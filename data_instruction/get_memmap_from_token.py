import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing
import random
import json

# Global config
MAX_SEQ_LEN = 791

def init_worker(transcriptions_, code_root_, tokenizer_, max_seq_len_, base_num_):
    global transcriptions, code_root, tokenizer, max_seq_len, base_num
    transcriptions = transcriptions_
    code_root = code_root_
    tokenizer = tokenizer_
    max_seq_len = max_seq_len_
    base_num = base_num_

def process_audio_id(audio_id):
    transcript, label = transcriptions.get(audio_id)
    if transcript is None:
        return None   

 
    code_file = os.path.join(code_root, audio_id + '.wav.npy')
    if not os.path.exists(code_file):
        return None
    try:
        codes = np.load(code_file, allow_pickle=True)
    except Exception as e:
        print(f"load {code_file}: {e}")
        return None
    
    codes = codes.squeeze().squeeze()
    if codes.ndim != 1:
        print(f"The shape of {code_file} is wrong: {codes.ndim}")
        return None

    codes = 128264 + torch.tensor(codes, dtype=torch.long)
 
    text_with_special = f"<|TEXT_UNDERSTANDING_START|>{transcript}<|TEXT_UNDERSTANDING_END|>"
    encoded_text = tokenizer.encode_plus(
        text_with_special,
        add_special_tokens=False,
        return_tensors='np'
    )
    text_input_ids = encoded_text['input_ids'].squeeze(0)

    speech_gen_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
    speech_gen_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
    code_input_ids = np.array(
        [speech_gen_start_id] + codes.tolist() + [speech_gen_end_id],
        dtype=np.int32
    )

    answer_with_special = f"<|ANSWER|>{label}"
    encoded_label = tokenizer.encode_plus(
        answer_with_special,
        add_special_tokens=False,
        return_tensors='np'
    )
    label_ids_with_eos = np.append(encoded_label['input_ids'].squeeze(0), tokenizer.eos_token_id).astype(np.int32)
    total_input_ids = np.concatenate([text_input_ids, code_input_ids, label_ids_with_eos])

    if len(total_input_ids) > max_seq_len:
        return None  # Skip too long

    # Pad
    total_input_ids = np.pad(
        total_input_ids,
        (0, max_seq_len - len(total_input_ids)),
        'constant',
        constant_values=tokenizer.pad_token_id
    )

    return total_input_ids.astype(np.int32), len(text_input_ids) + len(code_input_ids) + len(label_ids_with_eos), audio_id


def process_data(transcriptions, code_root, output_dir_tts, num_processes=4):
    tokenizer = AutoTokenizer.from_pretrained(
        'HKUSTAudio/Llasa-1B',
        model_max_length=MAX_SEQ_LEN,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = [
        '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
        '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
        '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
        '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>',
        '<|ANSWER|>'
    ]
    tokenizer.add_tokens(special_tokens)
    base_num = len(tokenizer)
    init_worker(transcriptions, code_root, tokenizer, MAX_SEQ_LEN, base_num)

    yes_ids = [aid for aid, (_, label) in transcriptions.items() if label == "yes"]
    no_ids = [aid for aid, (_, label) in transcriptions.items() if label == "no"]

    random.shuffle(yes_ids)
    random.shuffle(no_ids)

    # Validation selection
    def get_valid_subset(audio_ids, target_count):
        selected_ids = []
        processed_data = []
        i = 0
        while len(selected_ids) < target_count and i < len(audio_ids):
            res = process_audio_id(audio_ids[i])
            if res is not None:
                selected_ids.append(audio_ids[i])
                processed_data.append((res[0], res[1]))
            i += 1
        if len(selected_ids) < target_count:
            raise RuntimeError(f"Only {len(selected_ids)} samples found for one class, need {target_count}")
        return selected_ids, processed_data

    print("Selecting validation set...")
    val_yes_ids, val_yes_data = get_valid_subset(yes_ids, 500)
    val_no_ids, val_no_data = get_valid_subset(no_ids, 500)

    val_audio_ids = val_yes_ids + val_no_ids
    val_tts_input_ids_list = [x[0] for x in val_yes_data + val_no_data]
    val_lengths = [x[1] for x in val_yes_data + val_no_data]

    val_audio_ids_set = set(val_audio_ids)

    # Remaining for training
    remaining_yes = [x for x in yes_ids if x not in val_audio_ids_set]
    remaining_no = [x for x in no_ids if x not in val_audio_ids_set]
    train_audio_ids = remaining_yes + remaining_no
    random.shuffle(train_audio_ids)

    # Multiprocessing for training
    print("Processing training set...")
    with multiprocessing.Pool(
        min(num_processes, multiprocessing.cpu_count()),
        initializer=init_worker,
        initargs=(transcriptions, code_root, tokenizer, MAX_SEQ_LEN, base_num)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_audio_id, train_audio_ids),
            total=len(train_audio_ids),
            desc="data processing"
        ))

    train_tts_input_ids_list = [res[0] for res in results if res is not None]
    train_ids = [res[2] for res in results if res is not None]

    train_lengths = [res[1] for res in results if res is not None]

    if not (train_tts_input_ids_list and val_tts_input_ids_list):
        print("No usable data found. Exiting.")
        return

    max_total_token_len = max(train_lengths + val_lengths)
    print(f"Max total token length (before pad): {max_total_token_len}")

    os.makedirs(output_dir_tts, exist_ok=True)

    train_arr = np.array(train_tts_input_ids_list)
    val_arr = np.array(val_tts_input_ids_list)
  

    # Save memmaps
    np.memmap(os.path.join(output_dir_tts, 'train_input_ids_diff.memmap'),
              dtype='int32', mode='w+', shape=train_arr.shape)[:] = train_arr
    np.memmap(os.path.join(output_dir_tts, 'val_input_ids_diff.memmap'),
              dtype='int32', mode='w+', shape=val_arr.shape)[:] = val_arr
    with open(os.path.join(output_dir_tts, 'val_audio_ids_diff.json'), 'w') as f:
        json.dump(val_audio_ids, f)
    with open(os.path.join(output_dir_tts, 'train_audio_ids_diff.json'), 'w') as f:
        json.dump(train_ids, f)
    

    np.save(os.path.join(output_dir_tts, 'train_input_ids_shape_diff.npy'), train_arr.shape)
    np.save(os.path.join(output_dir_tts, 'val_input_ids_shape_diff.npy'), val_arr.shape)

    print(f"Train: {train_arr.shape}, Val: {val_arr.shape}")
    print("TTS memmaps saved to", output_dir_tts)


if __name__ == "__main__":
    code_root = '../xcodec_2/vq_codes'
    trans_file = '../xcodec_2/output_diff.csv'
    output_dir_tts = '../xcodec_2'
    num_processes = 8

    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            audio_id, transcript, label = parts[0], parts[1], parts[2]
            transcriptions[audio_id] = [transcript, label]

    process_data(transcriptions, code_root, output_dir_tts, num_processes)
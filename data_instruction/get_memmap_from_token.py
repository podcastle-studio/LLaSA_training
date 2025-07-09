 
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing
import random

dict_len = {}
 
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
    text_input_ids = encoded_text['input_ids'].squeeze(0)  # (text_len,)
 
    speech_gen_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
    speech_gen_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
    code_input_ids = np.array(
        [speech_gen_start_id] +
        codes.tolist() +
        [speech_gen_end_id],
        dtype=np.int32
    )

    answer_with_special = f"<|ANSWER|>{label}"#tokenizer.convert_tokens_to_ids('<|ANSWER|>')
    encoded_label = tokenizer.encode_plus(
        answer_with_special,
        add_special_tokens=False,
        return_tensors='np'
    )
    label_ids_with_eos = np.append(encoded_label['input_ids'].squeeze(0), tokenizer.eos_token_id).astype(np.int32)
 
    total_input_ids = np.concatenate([text_input_ids, code_input_ids, label_ids_with_eos])

    dict_len[audio_id]= len(total_input_ids)
    if len(total_input_ids) > max_seq_len:
        total_input_ids = total_input_ids[:max_seq_len]
        return None
        #print("", audio_id, "exceeds max_seq_len, truncated to", max_seq_len)
    else:
        padding_length = max_seq_len - len(total_input_ids)
        total_input_ids = np.pad(
            total_input_ids,
            (0, padding_length),
            'constant',
            constant_values=tokenizer.pad_token_id
        )

    return total_input_ids.astype(np.int32), len(text_input_ids) + len(code_input_ids) + len(label_ids_with_eos)


def process_data(transcriptions, code_root, output_dir_tts, num_processes=4):
    max_seq_len = 2532
 
    tokenizer = AutoTokenizer.from_pretrained(
        'HKUSTAudio/Llasa-1B',
        model_max_length=max_seq_len,
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
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
 
    base_num = len(tokenizer)

 
    audio_ids = list(transcriptions.keys())

    yes_ids = [aid for aid in audio_ids if transcriptions[aid][1] == "yes"]
    no_ids = [aid for aid in audio_ids if transcriptions[aid][1] == "no"]

    # Check you have enough
    assert len(yes_ids) >= 500, f"Only {len(yes_ids)} 'yes' samples available"
    assert len(no_ids) >= 500, f"Only {len(no_ids)} 'no' samples available"

    # Step 2: Shuffle for randomness
    random.shuffle(yes_ids)
    random.shuffle(no_ids)

    # Step 3: Create test set
    test_yes = yes_ids[:500]
    test_no = no_ids[:500]
    val_audio_ids = test_yes + test_no
    random.shuffle(val_audio_ids)

    # Step 4: Remaining = training
    remaining_yes = yes_ids[500:]
    remaining_no = no_ids[500:]
    train_audio_ids = remaining_yes + remaining_no
    random.shuffle(train_audio_ids)



    num_processes = min(num_processes, multiprocessing.cpu_count())

 
    with multiprocessing.Pool(
        num_processes,
        initializer=init_worker,
        initargs=(transcriptions, code_root, tokenizer, max_seq_len, base_num)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_audio_id, train_audio_ids),
            total=len(train_audio_ids),
            desc="data processing"
        ))
    train_tts_input_ids_list = [res[0] for res in results if res is not None]
    train_lengths = [res[1] for res in results if res is not None]

    val_lengths = []
    init_worker(transcriptions, code_root, tokenizer, max_seq_len, base_num)
    val_tts_input_ids_list = []
    for audio_id in tqdm(val_audio_ids, desc="valid data processing"):
        res = process_audio_id(audio_id)
        if res is not None:
            val_tts_input_ids_list.append(res[0])
            val_lengths.append(res[1])
 
    if not (train_tts_input_ids_list or val_tts_input_ids_list):
        print("bug ")
        return

    max_true_length = max(train_lengths + val_lengths)
    
    all_ids = train_tts_input_ids_list + val_tts_input_ids_list
    max_total_token_len = max(len(ids) for ids in all_ids)
 
    print("max token ",max_true_length)
    os.makedirs(output_dir_tts, exist_ok=True)

 
    train_tts_input_ids_array = np.array(train_tts_input_ids_list)
    val_tts_input_ids_array = np.array(val_tts_input_ids_list)

    train_tts_memmap_path = os.path.join(output_dir_tts, 'train_input_ids.memmap')
    train_tts_memmap = np.memmap(
        train_tts_memmap_path, dtype='int32', mode='w+', shape=train_tts_input_ids_array.shape
    )
    train_tts_memmap[:] = train_tts_input_ids_array[:]
    del train_tts_memmap   

    val_tts_memmap_path = os.path.join(output_dir_tts, 'val_input_ids.memmap')
    val_tts_memmap = np.memmap(
        val_tts_memmap_path, dtype='int32', mode='w+', shape=val_tts_input_ids_array.shape
    )
    val_tts_memmap[:] = val_tts_input_ids_array[:]
    del val_tts_memmap   

 
    np.save(os.path.join(output_dir_tts, 'train_input_ids_shape.npy'), train_tts_input_ids_array.shape)
    np.save(os.path.join(output_dir_tts, 'val_input_ids_shape.npy'), val_tts_input_ids_array.shape)

    print(f" TTS memmap  saved ! {output_dir_tts}")

if __name__ == "__main__":
 
    code_root = '../xcodec_2/vq_codes'
    
 
    # LJ001-0001|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition|...
    trans_file = '../xcodec_2/output.csv'
    
    output_dir_tts = '../xcodec_2'
 
    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 2:
                continue
            audio_id = parts[0]
            transcript = parts[1]   
            label = parts[2]
            transcriptions[audio_id] =[transcript, label]
 
    num_processes = 8

 
    process_data(
        transcriptions,
        code_root,
        output_dir_tts,
        num_processes=num_processes
    )
    print(max(dict_len.values()))

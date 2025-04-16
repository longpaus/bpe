from tokenizer.bpe import run_train_bpe
import pickle

if __name__ == '__main__':
    results = run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 10000, ['<|endoftext|>'])
    results_save_path = '../models/tokenizer/valid_results.pkl'

    try:
        with open(results_save_path, 'wb') as f: # 'wb' mode is crucial for pickle (write binary)
            pickle.dump(results, f)
    except Exception as e:
        print(f"Error saving with pickle: {e}")


num_sents = 10_000_000

wiki_file = "/mnt/nlu/users/andreas_hanselowski/models/mlffp/bglm_decom/data/wikipedia/enwiki_2023-09-01_sentences.txt"
out_file = f"data/words_num_sents_{num_sents}.txt"

with open(wiki_file, "r") as f:
    for i, line in enumerate(f):
        if i >= num_sents:
            break
        with open(out_file, "a") as out:
            # convert to lower case split line into words
            words = line.lower().split()

            # clean up the words, remove words with 1 character, remove words with numbers, remove words with special characters
            words = [word for word in words if len(word) > 1 and not any(char.isdigit() for char in word) and word.isalnum()]

            # write the words to the output file
            out.write(" ".join(words) + "\n")





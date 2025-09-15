import tiktoken
from sequence_generator import SequenceData


data = SequenceData()
N = 1

encoding = tiktoken.get_encoding("cl100k_base")

for i in range(N):
    seq = data.generate_sequence()
    # print(f"Generated sequence {i}: {seq.numpy()}")
    test_seq = seq.numpy()[:3]
    enc = encoding.encode(','.join(map(str, seq.numpy())))
    test_enc = encoding.encode(','.join(map(str, test_seq)))
    val = seq.numpy()[3]
    val_enc = encoding.encode(str(val))
    if len(enc) != 15:
        print(f"Sequence {i} of length {len(enc)}: {seq.numpy()}")
    if len(test_enc) != 11:
        print(f"Test sequence {i} of length {len(test_enc)}: {test_seq}")
    if len(val_enc) != 3:
        print(f"Value {i} of length {len(val_enc)}: {val}")


print(seq.numpy())
print(','.join(map(str, seq.numpy())))
print(enc)
print(test_enc)
print(val_enc)

print(encoding.decode([25741]))
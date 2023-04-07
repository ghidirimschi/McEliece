import random

import numpy as np

H = np.array([[1, 0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]])

G = np.array([[1, 1, 0, 1],
              [1, 0, 1, 1],
              [1, 0, 0, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = np.array([[0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]])


def detect_error(err_enc_bits):
    err_idx_vec = np.mod(H.dot(err_enc_bits), 2)
    err_idx_vec = err_idx_vec[::-1]
    err_idx = int(''.join(str(bit) for bit in err_idx_vec), 2)
    return err_idx - 1


def hamming7_4_encode(p_str):
    p = np.array([int(x) for x in p_str])

    prod = np.mod(G.dot(p), 2)
    return prod


def hamming7_4_decode(c):
    prod = np.mod(R.dot(c), 2)
    return prod


def flip_bit(bits, n):
    bits[n] = (bits[n] + 1) % 2


def add_single_bit_error(enc_bits):
    error = [0] * 7
    idx = random.randint(0, 6)
    error[idx] = 1
    return np.mod(enc_bits + error, 2)


def split_binary_string(str, n):
    return [str[i:i + n] for i in range(0, len(str), n)]


def bits_to_str(bits):
    # Split the binary string into 8-bit chunks
    my_chunks = [bits[i:i + 8] for i in range(0, len(bits), 8)]

    # Convert each 8-bit chunk to its corresponding character
    my_chars = [chr(int(chunk, 2)) for chunk in my_chunks]

    # Concatenate the characters into a single string
    my_text = ''.join(my_chars)

    # Print the resulting text
    return my_text


if __name__ == '__main__':
    input = input('Provide the string to encode: ')
    binary_str = ''.join(format(ord(x), '08b') for x in input)
    print("Binary representation of the input: ")
    print(binary_str)

    # split bits into chunks of 4
    split_bits_list = split_binary_string(binary_str, 4)
    enc_msg = []
    for split_bits in split_bits_list:
        # encode using hamming7-4
        enc_bits = hamming7_4_encode(split_bits)
        # add a random bit error
        err_enc_bits = add_single_bit_error(enc_bits)

        # convert to string and append to result
        str_enc = ''.join(str(x) for x in err_enc_bits)
        enc_msg.append(str_enc)

    print("Binary representation of the encoded message: ")
    print(''.join(enc_msg))

    dec_msg = []
    for enc_bits in enc_msg:
        # convert to ints
        enc_bits = [int(x) for x in enc_bits]
        # detect the error position
        err_idx = detect_error(enc_bits)
        # correct the error
        flip_bit(enc_bits, err_idx)
        # decode fixed bits
        out = hamming7_4_decode(enc_bits)

        str_dec = ''.join(str(x) for x in out)
        dec_msg.append(str_dec)

    dec_msg_str = ''.join(dec_msg)
    print("Binary representation of the decoded message: ")
    print(dec_msg_str)
    txt = bits_to_str(dec_msg_str)
    print('Decoded result:')
    print(txt)

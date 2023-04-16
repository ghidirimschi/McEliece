import random
import numpy as np
from constraint import *

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


def random_binary_non_singular_matrix(n):
    a = np.random.randint(0, 2, size=(n, n))
    while (np.linalg.det(a) % 2) == 0:
        a = np.random.randint(0, 2, size=(n, n))
    return a


S = random_binary_non_singular_matrix(4)
S_inv = np.mod(np.linalg.inv(S).astype(int), 2)


def generate_permutation_matrix(n):
    i = np.eye(n)
    p = np.random.permutation(i)
    return p.astype(int)


P = generate_permutation_matrix(7)
P_inv = np.mod(np.linalg.inv(P).astype(int), 2)

G_hat = np.transpose(np.mod((S.dot(np.transpose(G))).dot(P), 2))


def detect_error(err_enc_bits):
    err_idx_vec = np.mod(H.dot(err_enc_bits), 2)
    err_idx_vec = err_idx_vec[::-1]
    err_idx = int(''.join(str(bit) for bit in err_idx_vec), 2)
    return err_idx - 1


def hamming7_4_encode(p_str):
    p = np.array([int(x) for x in p_str])

    prod = np.mod(G_hat.dot(p), 2)
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


# solve G_hat*m=enc, returns random solution if more than 1 or None if none
def CSP_solve_matrix(G_hat, enc):
    problem = Problem()
    problem.addVariables(['a', 'b', 'c', 'd'], range(0, 2))

    def constraint_func(a, b, c, d):
        nparr = np.array([a, b, c, d])
        return np.array_equal(enc, np.mod(G_hat.dot(nparr), 2))

    problem.addConstraint(constraint_func,
                          ['a', 'b', 'c', 'd'])

    sols = problem.getSolutions()

    if len(sols) == 0:
        return None

    idx = random.randint(0, len(sols) - 1)
    sol = sols[idx]
    vec = [sol['a'], sol['b'], sol['c'], sol['d']]
    vec = np.array(vec)
    return vec


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
        enc_bits = np.array([int(x) for x in enc_bits])
        # guess error
        for i in range(0, 7):
            try_error = [0] * 7
            try_error[i] = 1
            try_enc_bits = np.mod(enc_bits + try_error, 2)
            try_solve = CSP_solve_matrix(G_hat, try_enc_bits)
            if try_solve is not None:
                str_dec = ''.join(str(x) for x in try_solve)
                dec_msg.append(str_dec)
                break

    dec_msg_str = ''.join(dec_msg)
    print("Binary representation of the broken message: ")
    print(dec_msg_str)
    txt = bits_to_str(dec_msg_str)
    print('Broken result:')
    print(txt)

    dec_msg = []
    for enc_bits in enc_msg:
        enc_bits = np.array([int(x) for x in enc_bits])
        # compute c_hat = c * P_inv
        c_hat = np.mod(enc_bits.dot(P_inv), 2)
        # find the error bit
        err_idx = detect_error(c_hat)
        # flip it
        flip_bit(c_hat, err_idx)
        # find m_hat
        m_hat = hamming7_4_decode(c_hat)
        # find m = m_hat * S_inv
        m_out = np.mod(m_hat.dot(S_inv), 2)

        str_dec = ''.join(str(x) for x in m_out)
        dec_msg.append(str_dec)

    dec_msg_str = ''.join(dec_msg)
    print("Binary representation of the decoded message: ")
    print(dec_msg_str)
    txt = bits_to_str(dec_msg_str)
    print('Decoded result:')
    print(txt)

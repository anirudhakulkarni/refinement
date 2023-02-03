import threading
import sys
from pyaes import AESModeOfOperationCTR


def bruteForce_16(partial_key, plain_data, enc_data):
    '''brute force the key with 16 bits random for the first plaintext

    Args:
        partial_key (bytes): partial key known 
        plain_data (bytes): plaintext
        enc_data (bytes): ciphertext
    Returns:
        key (bytes): encryption key which is 16 bytes long
    '''

    # key should be 16 bytes long
    # current key is 14 bytes long
    # iterate over all bytes possible and append to key and check if the key is correct
    for i in range(256):
        for j in range(256):
            key = partial_key+ bytes([i])+bytes([j])
            cipher = AESModeOfOperationCTR(key)
            decrypted = cipher.decrypt(plain_data)
            if decrypted==enc_data:
                print('key is: ', key)
                return key
# runtime: 6 seconds


def bruteForce_32(partial_key, plain_data, enc_data):
    '''brute force the key with 32 bits random for the second plaintext
    
    Args:
        partial_key (bytes): partial key known
        plain_data (bytes): plaintext
        enc_data (bytes): ciphertext    
    Returns:
        key (bytes): encryption key
    '''
    
    
    def try_key(i,j):
        '''try key with 32 bits random for the second plaintext

        Args:
            i (int): first byte
            j (int): second byte
        Returns:
            key (bytes): encryption key
        '''
        for k in range(0, 256):
            for l in range(0, 256):
                key = partial_key+bytes([i, j, k, l])
                cipher = AESModeOfOperationCTR(key)
                if cipher.decrypt(plain_data)==enc_data:
                    print('found key:', key)
                    return key
    # parallely try all keys and if found return key
    my_threads = []
    for i in range(0, 256):
        for j in range(0, 256):
            # launch a thread
            t = threading.Thread(target=try_key, args=(i,j))
            t.start()
            my_threads.append(t)
    for t in my_threads:
        if t.join():
            return t.join()

# runtime: 10864 seconds (3 hours) on 64 cores

def write_key(key):
    '''write key to file

    Args:
        bits (int): 16, 32 or 48
        key (bytes): encryption key
    '''
    try:
        with open("Key.txt", "a") as file1:
            file1.writelines(str(key))
    except:
        with open("Key.txt", "w") as file1:
            file1.writelines(str(key))
    return

def main():
    plaintexts = [b'First plaintext', b'Second plaintext']
    ciphertexts = [b'\x8b]\x13\r\x9c\x80o`1?W\xc4\x9df\xbd', b'\xec:\x8c\xd6)\x04\xe8<\x8b\xcb\x06\xab\xd6\xf0n\x00']

    # 16 bits unknown
    plain_data = plaintexts[0]
    enc_data = ciphertexts[0]
    partial_key = b'\x00'*14
    key_16 = bruteForce_16(partial_key, plain_data, enc_data)
    print('key is: ', key_16)

    # 32 bits unknown
    plain_data = plaintexts[1]
    enc_data = ciphertexts[1]
    partial_key = b'\x00'*12
    key_32 = bruteForce_32(partial_key, plain_data, enc_data)
    print('key is: ', key_32)

    #  write keys to file
    write_key(key_16)
    write_key(key_32)

if __name__ == '__main__':
    main()




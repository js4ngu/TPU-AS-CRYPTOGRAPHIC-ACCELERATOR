import torch
import json

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_gcd(b % a, a)
        return (g, y - (b // a) * x, x)

def modinv(a, m):
    g, x, y = extended_gcd(a, m)
    return x % m


def nttMatrix_optV3(w, N, p):
    print("def NTT")
    W = torch.zeros((N, N), dtype=torch.int32)
    powers = [1]
    for _ in range(N*N):
        print("NOW Calc W : ", (_ / (N*N))*100)
        powers.append((powers[-1] * w) % p)

    for i in range(N):
        print("NOW Define NTT mat : ", (i/N)*100)
        for j in range(i, N):
            W[i, j] = powers[i*j]
            W[j, i] = W[i, j]
    return W

def invNttMatrix_optV3(w, N, p):
    print("def INTT")
    W = torch.zeros((N, N), dtype=torch.int32)
    invW = modinv(w, p)
    invN = modinv(N, p)
    
    powers_invW = [1]
    for _ in range(N*N):
        print("NOW Calc invW : ", (_ / (N*N))*100)
        powers_invW.append((powers_invW[-1] * invW) % p)

    for i in range(N):
        print("NOW Define invNTT mat", (i/N)*100)
        for j in range(i, N):
            val = powers_invW[i*j]
            W[i, j] = (invN * val) % p
            W[j, i] = W[i, j]
    return W

def exportMatrix(w, N, p, file_path):
    invW = modinv(w, p)
    invN = modinv(N, p)

    nttMat = nttMatrix_optV3(w, N, p)
    inttMat = invNttMatrix_optV3(w, N, p)

    data = {
        "NTT": nttMat.tolist(),
        "INTT": inttMat.tolist(),
        "w": w,
        "N": N,
        "invW": invW,
        "invN": invN,
        "p": p
    }
    with open(file_path, 'w') as f:
        print("now dump!")
        json.dump(data, f)

def readJson(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def LoadParameter(file_path, matrix_type):
    data = readJson(file_path)
    
    if matrix_type in ['NTT', 'INTT', 'N', 'w', 'invW', 'invN', 'p']:
        return data.get(matrix_type)
    else:
        raise ValueError("Invalid matrix_type. Choose among 'NTT', 'INTT', 'N', 'w', 'invW', or 'invN'.")

def main():
    # JSON 파일 경로pt
    json_file_path = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/FHE/rns11.json'  # JSON 파일 경로를 여기에 삽입하세요.
    
    # JSON 파일 로드
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # w 값과 Dec 값을 추출합니다.
    w = [entry['w'] for entry in data]
    moduli = [entry['Dec'] for entry in data]
    
    N = 2**11  # 이 값은 문제에 따라 조정될 수 있습니다.

    for i in range(len(w)):
        print("now def :", i/len(w) * 100, "%")
        file_path = "/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/parameter/PWC/FHE/11/"+str(moduli[i]) + ".json"
        print(file_path)
        exportMatrix(w[i], N, moduli[i], file_path)

# main 함수 호출
main()

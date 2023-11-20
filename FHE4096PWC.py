from ntt_tpu_operations import rnsNTTbasedPolymult, RandomPoly, paddedRandomPoly
import json

def main():
    #Enviroment parameter
    degreeSize = 2**12
    coefBit = (52, 52) #50까지 pass
    peSize = 64
    rns_ch_list = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/FHE/rns12.json'  # JSON 파일 경로를 여기에 삽입하세요.
    
    # JSON 파일 로드
    with open(rns_ch_list, 'r') as f:
        data = json.load(f)
    
    # w 값과 Dec 값을 추출합니다.
    w = [entry['w'] for entry in data]
    #psi = [entry['psi'] for entry in data]
    moduli = [entry['Dec'] for entry in data]
    
    parameterDir = "/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/parameter/PWC/FHE/12/"
    a = RandomPoly(degreeSize, coefBit) #[1,2,3,4]
    b = RandomPoly(degreeSize, coefBit) #[5,6,7,8]
    p = 2**116 - 2**18 + 1
    mode = "positive"
    #ntt based polymult calc
    rnsNTTbasedPolymult(a, b, p, mode, w, moduli, degreeSize, peSize, parameterDir)
    print(coefBit)
main()
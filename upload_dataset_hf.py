"""
py -3.10 -m uv venv venv

.\venv\Scripts\Activate.ps1

윈도우에서 CUDA 버전 확인 방법
nvcc --version

GPU 드라이버가 지원하는 최대 CUDA 버전
nvidia-smi

uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
"""

from datasets import load_dataset
from huggingface_hub import login

# 1. 로그인 (아까 복사한 Write 토큰을 따옴표 안에 넣으세요)
login(token="")

# 2. 로컬에 있는 JSON 파일 불러오기
# split="train"을 넣어줘야 나중에 불러올 때 편합니다.
dataset = load_dataset("json", data_files="mydata1_gemma3_1b_it.json", split="train")

# 3. 허깅페이스에 업로드 (Private 설정 필수!)
# "본인아이디/데이터셋이름" 형식으로 적어주세요.
# 예: "gildong/company-manual-v1"
my_repo_id = "WildOjisan/gemma3_1b_it_sample" 

dataset.push_to_hub(my_repo_id, private=False) 

print(f"업로드 완료! 주소: https://huggingface.co/datasets/{my_repo_id}")
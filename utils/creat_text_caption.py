import os
import json
import argparse
import glob
from PIL import Image
from google import genai
from google.genai import types
import yaml
import time

def load_api_key():
    """API 키를 vault/key.yaml 파일에서 로드합니다."""
    try:
        with open('vault/key.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config.get('google_api_key')
    except Exception as e:
        print(f"API 키 로드 중 오류 발생: {str(e)}")
        return None

def generate_caption(image_path, prompt, temperature=0.9):
    """
    Gemini를 사용하여 이미지에 대한 캡션을 생성합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        prompt (str): 캡션 생성을 위한 프롬프트
        temperature (float): 창의성 수준 (0.0-1.0)
        
    Returns:
        str: 생성된 캡션
    """
    try:
        # 이미지 파일 열기
        image = Image.open(image_path)
        
        # Gemini API 호출
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=2048,
                temperature=temperature
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"캡션 생성 중 오류 발생: {str(e)}")
        return None

def process_images(image_dir, prompt, output_file="captions.json", temperature=0.9):
    """
    이미지 폴더의 모든 이미지에 대해 캡션을 생성하고 JSON 파일로 저장합니다.
    
    Args:
        image_dir (str): 이미지가 저장된 디렉토리 경로
        prompt (str): 캡션 생성을 위한 프롬프트
        output_file (str): 결과를 저장할 JSON 파일 경로
        temperature (float): 창의성 수준 (0.0-1.0)
    """
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        pattern = os.path.join(image_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"'{image_dir}' 디렉토리에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")
    
    # 결과 저장용 딕셔너리
    captions = {}
    
    # 각 이미지 처리
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(image_files)}] 처리 중: {filename}")
        
        try:
            # 캡션 생성
            caption = generate_caption(image_path, prompt, temperature)
            
            if caption:
                # 결과 저장
                captions[filename] = caption
                print(f"✓ 성공: {caption[:100]}...")
            else:
                print(f"✗ 실패: 캡션 생성 실패")
            
            # API 호출 간 딜레이
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ 오류 발생: {str(e)}")
    
    # 결과를 JSON 파일로 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        print(f"\n캡션이 {output_file}에 저장되었습니다.")
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='이미지에 대한 텍스트 캡션을 생성하고 JSON 파일로 저장합니다.')
    parser.add_argument('--image-dir', type=str, 
                        default='ghibli_2',
                        help='이미지가 저장된 디렉토리 경로')
    parser.add_argument('--prompt', type=str,
                        default="Describe the input image in 1–2 sentences. Make sure to mention that the image is in 'Ghibli style'.",
                        help='캡션 생성을 위한 프롬프트')
    parser.add_argument('--output', type=str, default='ghibli_2_captions.json', help='결과를 저장할 JSON 파일 경로')
    parser.add_argument('--temperature', type=float, default=0.5, help='창의성 수준 (0.0-1.0)')
    
    args = parser.parse_args()
    
    # API 키 로드
    global GOOGLE_API_KEY, genai_client
    GOOGLE_API_KEY = load_api_key()
    if not GOOGLE_API_KEY:
        raise ValueError("Google API 키를 찾을 수 없습니다. vault/key.yaml 파일을 확인해주세요.")
    
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # 이미지 처리 시작
    print("\n작업 설정:")
    print(f"  - 이미지 디렉토리: {args.image_dir}")
    print(f"  - 프롬프트: {args.prompt}")
    print(f"  - 출력 파일: {args.output}")
    print(f"  - 창의성 수준(temperature): {args.temperature}")
    
    process_images(
        image_dir=args.image_dir,
        prompt=args.prompt,
        output_file=args.output,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding=utf-8

"""
두 개의 캡션 JSON 파일을 병합하는 스크립트

사용법:
python merge_captions.py --file1 ghibli_captions.json --file2 ghibli_2_captions.json --output merged_captions.json
"""

import argparse
import json
import os

def load_json_file(file_path):
    """JSON 파일을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"파일 로드 중 오류 발생 ({file_path}): {str(e)}")
        return None

def merge_captions(file1_path, file2_path, output_path):
    """
    두 개의 캡션 JSON 파일을 병합합니다.
    
    Args:
        file1_path (str): 첫 번째 JSON 파일 경로
        file2_path (str): 두 번째 JSON 파일 경로
        output_path (str): 병합된 결과를 저장할 파일 경로
    """
    # JSON 파일 로드
    captions1 = load_json_file(file1_path)
    captions2 = load_json_file(file2_path)
    
    if captions1 is None or captions2 is None:
        return
    
    # 두 딕셔너리 병합
    merged_captions = {**captions1, **captions2}
    
    # 중복된 키 확인
    duplicates = set(captions1.keys()) & set(captions2.keys())
    if duplicates:
        print(f"\n경고: {len(duplicates)}개의 중복된 이미지가 발견되었습니다:")
        for key in duplicates:
            print(f"  - {key}")
    
    # 결과 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_captions, f, indent=2, ensure_ascii=False)
        print(f"\n병합 완료:")
        print(f"  - 파일 1: {len(captions1)}개 캡션")
        print(f"  - 파일 2: {len(captions2)}개 캡션")
        print(f"  - 병합 결과: {len(merged_captions)}개 캡션")
        print(f"  - 저장 위치: {output_path}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='두 개의 캡션 JSON 파일을 병합합니다.')
    parser.add_argument('--file1', type=str,
                        default='ghibli/ghibli_captions_simplified.json',
                        help='첫 번째 JSON 파일 경로')
    parser.add_argument('--file2', type=str, 
                        default='ghibli_2_captions.json',
                        help='두 번째 JSON 파일 경로')
    parser.add_argument('--output', type=str,
                        default='ghibli/merged_captions.json',
                        help='병합된 결과를 저장할 파일 경로')
    
    args = parser.parse_args()
    
    # 파일 존재 여부 확인
    if not os.path.exists(args.file1):
        print(f"오류: 파일을 찾을 수 없습니다: {args.file1}")
        return
    if not os.path.exists(args.file2):
        print(f"오류: 파일을 찾을 수 없습니다: {args.file2}")
        return
    
    # 병합 실행
    merge_captions(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main() 
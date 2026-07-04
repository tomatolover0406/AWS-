"""
실습 7 (Bedrock 버전): 이미지 분류 & 캡셔닝 on Amazon Bedrock
============================================================


장점
---------------------
- 클러스터같은 "켜놓고 잊어버리는" 상시 비용이 없음. Bedrock은 서버리스라 호출한 토큰만큼만 과금됨 → idle 비용 0.
- 분류(ViT)와 캡셔닝을 모두 멀티모달 모델 한 개(Amazon Nova Lite)로 처리.
- 더 높은 품질이 필요하면 MODEL_ID만 Claude Haiku 등으로 변경.

사전 준비
---------
    pip install boto3 pillow
    # AWS 자격증명 설정 (둘 중 하나)
    #   aws configure                # AWS CLI
    #   또는 환경변수 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
    # Bedrock 콘솔에서 해당 리전에 Nova Lite 모델 액세스를 활성화해 둘 것.

실행
----
    python lab07_bedrock_nova.py --image-dir ./images
    # ./images 안의 jpg/png 를 모두 분류 + 캡셔닝
"""

import argparse
import base64
import glob
import io
import json
import os

import boto3
from PIL import Image, ImageFile

# 손상/잘린 이미지 허용 (원본 Lab v5 수정과 동일 취지)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# 💰 비용 적절: Amazon Nova Lite (비전 지원, 매우 저렴)
#   - 리전에 따라 cross-region inference profile ID가 필요할 수 있음:
#     us.amazon.nova-lite-v1:0 (US), eu.amazon.nova-lite-v1:0 (EU) 등.
#   - 권한/리전 오류가 나면 "us." 접두사 버전으로 바꿔볼 것.
MODEL_ID = "amazon.nova-lite-v1:0"
# MODEL_ID = "us.amazon.nova-lite-v1:0"          # cross-region 필요 시
# MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"  # 고품질·비용↑ 대안

CAPTION_MAX_TOKENS = 128   # 출력 토큰 절감 (원본 200 → 128)
TEMPERATURE = 0.3

# 분류 후보 카테고리 (원본 Lab의 5종 음식). 자유롭게 수정 가능.
CATEGORIES = ["pizza", "sushi", "fried_rice", "ramen", "ice_cream"]

# 💰 Nova Lite 토큰 단가 (USD per 1,000,000 tokens) — 청구 시점 단가로 갱신할 것
PRICE_PER_1M_INPUT_USD = 0.06
PRICE_PER_1M_OUTPUT_USD = 0.24
USD_TO_KRW = 1380.0

# 세션 전체 토큰 누적기
TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}

# Bedrock Runtime 클라이언트
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심: Bedrock Converse API 호출 (이미지 + 텍스트)
# ─────────────────────────────────────────────────────────────────────────────
def _image_bytes(image: Image.Image) -> bytes:
    """PIL 이미지를 JPEG 바이트로 변환."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def call_vision(image: Image.Image, prompt: str, max_tokens: int = CAPTION_MAX_TOKENS) -> str:
    """이미지+프롬프트를 Bedrock 멀티모달 모델에 보내고 텍스트 응답을 반환.
    실제 사용 토큰을 TOKEN_USAGE 에 누적한다."""
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": _image_bytes(image)}}},
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"maxTokens": max_tokens, "temperature": TEMPERATURE},
    )

    # 토큰 사용량 누적 (Converse 응답은 usage 를 항상 반환)
    usage = response.get("usage", {})
    it = int(usage.get("inputTokens", 0) or 0)
    ot = int(usage.get("outputTokens", 0) or 0)
    tt = int(usage.get("totalTokens", it + ot) or (it + ot))
    TOKEN_USAGE["input_tokens"] += it
    TOKEN_USAGE["output_tokens"] += ot
    TOKEN_USAGE["total_tokens"] += tt
    TOKEN_USAGE["calls"] += 1

    # 응답 텍스트 추출
    return response["output"]["message"]["content"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Part A: 분류 — 멀티모달 모델로 카테고리 판별 (ViT 대체, Bedrock만으로 완결)
# ─────────────────────────────────────────────────────────────────────────────
def classify_image(image: Image.Image, categories=CATEGORIES) -> dict:
    """주어진 후보 중 가장 가까운 카테고리를 JSON으로 받는다."""
    cat_list = ", ".join(categories)
    prompt = (
        f"Classify this food image into exactly one of: {cat_list}. "
        "Respond ONLY as compact JSON: "
        '{"label": "<one_category>", "confidence": <0.0-1.0>}. '
        "No extra text."
    )
    raw = call_vision(image, prompt, max_tokens=40)
    try:
        # 모델이 코드펜스를 붙일 수 있어 정리
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
        return {"label": data.get("label", "unknown"), "confidence": float(data.get("confidence", 0.0))}
    except Exception:
        return {"label": raw[:30], "confidence": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Part B: 캡셔닝
# ─────────────────────────────────────────────────────────────────────────────
def caption_image(
    image: Image.Image,
    prompt: str = (
        "Describe this food image in detail. Include the type of food, "
        "its appearance, and likely ingredients."
    ),
) -> str:
    return call_vision(image, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 비용 리포트
# ─────────────────────────────────────────────────────────────────────────────
def print_cost_summary():
    it = TOKEN_USAGE["input_tokens"]
    ot = TOKEN_USAGE["output_tokens"]
    tt = TOKEN_USAGE["total_tokens"]
    calls = TOKEN_USAGE["calls"]

    input_cost = it / 1_000_000 * PRICE_PER_1M_INPUT_USD
    output_cost = ot / 1_000_000 * PRICE_PER_1M_OUTPUT_USD
    total_cost = input_cost + output_cost

    print("=" * 60)
    print(f"💰 세션 토큰 사용량 & 예상 비용 (Bedrock {MODEL_ID})")
    print("=" * 60)
    print(f"  호출 횟수   : {calls:,} 회")
    print(f"  입력 토큰   : {it:,}")
    print(f"  출력 토큰   : {ot:,}")
    print(f"  총 토큰     : {tt:,}")
    print("-" * 60)
    print(f"  입력 비용   : ${input_cost:.6f}")
    print(f"  출력 비용   : ${output_cost:.6f}")
    print(f"  ▶ 예상 총액 : ${total_cost:.6f}  (약 ₩{total_cost * USD_TO_KRW:,.2f})")
    print("=" * 60)
    if calls:
        print(f"  * 호출당 평균: {tt/calls:,.0f} 토큰, ${total_cost/calls:.6f}")
    print("  ※ 정확한 청구액은 AWS Cost Explorer / CloudWatch 로 확인하세요.")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────
def load_images(image_dir: str):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
        paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    paths = sorted(set(paths))
    images = []
    for p in paths:
        try:
            img = Image.open(p)
            img.load()
            images.append((os.path.basename(p), img.convert("RGB")))
        except Exception as e:
            print(f"  ⏭️  로드 실패 건너뜀: {p} ({e})")
    return images


def main():
    parser = argparse.ArgumentParser(description="Bedrock 이미지 분류 + 캡셔닝")
    parser.add_argument("--image-dir", default="./images", help="이미지 폴더 경로")
    parser.add_argument("--no-caption", action="store_true", help="캡셔닝 건너뛰기(분류만)")
    args = parser.parse_args()

    print(f"📌 리전: {REGION} / 모델: {MODEL_ID}")
    images = load_images(args.image_dir)
    if not images:
        print(f"⚠️  '{args.image_dir}' 에서 이미지를 찾지 못했습니다.")
        return
    print(f"✅ 이미지 {len(images)}장 로드 완료\n")

    print("🔍 분류 + 캡셔닝 결과")
    print("=" * 70)
    for name, img in images:
        cls = classify_image(img)
        line = f"\n📷 {name}\n   분류: {cls['label']} ({cls['confidence']:.0%})"
        if not args.no_caption:
            cap = caption_image(img)
            line += f"\n   캡션: {cap}"
        print(line)
        print("-" * 70)

    print()
    print_cost_summary()


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger

# 제품 이미지 데이터셋 클래스
class ProductImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.products_dir = self.dataset_dir / "products"
        self.styled_dir = self.dataset_dir / "styled"
        self.transform = transform
        
        logger.info(f"Loading dataset from {dataset_dir}")
        logger.info(f"Products directory: {self.products_dir}")
        logger.info(f"Styled directory: {self.styled_dir}")
        
        # 캡션 파일 로드
        captions_path = self.dataset_dir / "captions.json"
        logger.info(f"Loading captions from {captions_path}")
        
        with open(captions_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
            logger.info(f"Loaded {len(self.captions)} entries from captions.json")
        
        # 데이터셋 항목 생성
        self.items = []
        for image_id, data in self.captions.items():
            product_img = data["product"]
            prompt = data["prompt"]
            styled_img = f"{image_id}.png"
            
            # 파일이 실제로 존재하는지 확인
            product_path = self.products_dir / product_img
            styled_path = self.styled_dir / styled_img
            
            if product_path.exists() and styled_path.exists():
                self.items.append({
                    "image_id": image_id,
                    "product_image": str(product_path),
                    "styled_image": str(styled_path),
                    "prompt": prompt
                })
            else:
                logger.warning(f"Missing files for {image_id}:")
                if not product_path.exists():
                    logger.warning(f"  Product image not found: {product_path}")
                if not styled_path.exists():
                    logger.warning(f"  Styled image not found: {styled_path}")
        
        if len(self.items) == 0:
            raise ValueError(
                f"No valid image pairs found in {dataset_dir}. "
                "Please check your dataset structure and captions.json file. "
                "Make sure both product and styled images exist."
            )
        
        logger.info(f"Successfully loaded {len(self.items)} image pairs from {dataset_dir}")
        logger.info(f"First few items: {[item['image_id'] for item in self.items[:3]]}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        logger.debug(f"__getitem__ called with idx: {idx}, type: {type(idx)}")
        try:
            item = self.items[idx]
            logger.debug(f"Retrieved item: {item['image_id']}")
            
            # 이미지 로드 및 전처리
            product_image = Image.open(item["product_image"]).convert("RGB")
            styled_image = Image.open(item["styled_image"]).convert("RGB")
            
            if self.transform:
                product_image = self.transform(product_image)
                styled_image = self.transform(styled_image)
            
            result = {
                "image": styled_image,
                "product_image": product_image,
                "text": item["prompt"]
            }
            logger.debug(f"Successfully processed item {item['image_id']}")
            return result
        except Exception as e:
            logger.error(f"Error in __getitem__ for idx {idx}: {str(e)}")
            logger.error(f"Item content: {self.items[idx] if idx < len(self.items) else 'Index out of range'}")
            raise

# 지브리 데이터셋 클래스
class GhibliDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.transform = transform
        
        logger.info(f"Loading Ghibli dataset from {dataset_dir}")
        logger.info(f"Images directory: {self.images_dir}")
        
        # 캡션 파일 로드
        captions_path = self.dataset_dir / "captions.json"
        logger.info(f"Loading captions from {captions_path}")
        
        with open(captions_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
            logger.info(f"Loaded {len(self.captions)} entries from captions.json")
        
        # 데이터셋 항목 생성
        self.items = []
        for idx, (image_name, caption) in enumerate(self.captions.items()):
            image_id = idx
            # 파일이 실제로 존재하는지 확인
            image_path = self.images_dir / image_name
            
            if image_path.exists():
                self.items.append({
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "caption": caption
                })
            else:
                logger.warning(f"Image not found: {image_path}")
        
        if len(self.items) == 0:
            raise ValueError(
                f"No valid images found in {dataset_dir}. "
                "Please check your dataset structure and ghibli_captions_simplified.json file."
            )
        
        logger.info(f"Successfully loaded {len(self.items)} images from {dataset_dir}")
        logger.info(f"First few items: {[item['image_id'] for item in self.items[:3]]}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        logger.debug(f"__getitem__ called with idx: {idx}, type: {type(idx)}")
        try:
            item = self.items[idx]
            logger.debug(f"Retrieved item: {item['image_id']}")
            
            # 이미지 로드 및 전처리
            image = Image.open(item["image_path"]).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            result = {
                "image": image,
                "text": item["caption"]
            }
            logger.debug(f"Successfully processed item {item['image_id']}")
            return result
        except Exception as e:
            logger.error(f"Error in __getitem__ for idx {idx}: {str(e)}")
            logger.error(f"Item content: {self.items[idx] if idx < len(self.items) else 'Index out of range'}")
            raise
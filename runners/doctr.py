from shared import for_each_pdf, run_with_metrics


def main():
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True, det_arch="db_resnet50", reco_arch="crnn_vgg16_bn")

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            doc = DocumentFile.from_images(img_path)
            result = model(doc)
            text = result.render()
            pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                          "n_chars": len(text),
                          "text": text})
        return {"n_pages": len(pages),
                "total_chars": sum(p["n_chars"] for p in pages),
                "pages": pages}

    return {"engine": "docTR (db_resnet50 + crnn_vgg16_bn)", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("doctr", main)

from pycocotools.coco import COCO

class AnnotationLoader:
    def __init__(self, annotation_file):
        self.coco = COCO(annotation_file)

    def get_image_ids(self):
        return self.coco.getImgIds()

    def get_image_info(self, image_id):
        return self.coco.loadImgs(image_id)[0]

    def get_annotations_for_image(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        return self.coco.loadAnns(ann_ids)

    def get_category_names(self):
        return [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]

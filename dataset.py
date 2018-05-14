import pandas as pd
fields = [
'weekend',
' coverphoto',
' event',
' link',
' photo',
' question',
' status',
' swf',
' video',
'coverphoto_reactions',
'coverphoto_comments',
'coverphoto_shares',
'event_reactions',
'event_comments',
'event_shares',
'link_reactions',
'link_comments',
'link_shares',
'photo_reactions',
'photo_comments',
'photo_shares',
'question_reactions',
'question_comments',
'question_shares',
'status_reactions',
'status_comments',
'status_shares',
'swf_reactions',
'swf_comments',
'swf_shares',
'video_reactions',
'video_comments',
'video_shares'
]

_ORIG = [
    'reactions',
    'comments',
    'shares',
    'type'
]

_CATEGORIES = [
    ' coverphoto',
    ' event',
    ' link',
    ' photo',
    ' question',
    ' status',
    ' swf',
    ' video'
]

_CONTINIOUS = [
    'coverphoto_reactions',
    'coverphoto_comments',
    'coverphoto_shares',
    'event_reactions',
    'event_comments',
    'event_shares',
    'link_reactions',
    'link_comments',
    'link_shares',
    'photo_reactions',
    'photo_comments',
    'photo_shares',
    'question_reactions',
    'question_comments',
    'question_shares',
    'status_reactions',
    'status_comments',
    'status_shares',
    'swf_reactions',
    'swf_comments',
    'swf_shares',
    'video_reactions',
    'video_comments',
    'video_shares'
]

_FILENAMES = [
    'Dodge_result.csv',
    'Ferrari_result.csv',
    'Hyundai_result.csv',
    'Kia_result.csv',
    'Mercedes-Benz_result.csv',
    'Mini_result.csv',
    'Peugeot_result.csv',
    'Toyota_result.csv',
    'Volkswagen_result.csv'
]

_BRANDS = [
    'BMW_result.csv',
    'Dodge_result.csv',
    'Ferrari_result.csv',
    'Hyundai_result.csv',
    'Kia_result.csv',
    'Mercedes-Benz_result.csv',
    'Mini_result.csv',
    'Peugeot_result.csv',
    'Toyota_result.csv',
    'Volkswagen_result.csv'
]

class Datasets:
    def getAll(self):
        res = pd.read_csv('./datasets/BMW_result.csv')
        for filename in _FILENAMES:
            df = pd.read_csv('./datasets/' + filename)
            res = res.append(df)
        return res

    def getByBrands(self):
        res = {}
        for brand in _BRANDS:
            df = pd.read_csv('./datasets/'+brand)
            res[brand] = df
        return res

    def getAllOrig(self):
        return self.getAll()[_ORIG].as_matrix()

    def getAllCont(self):
        return self.getAll()[_CONTINIOUS].as_matrix()

    def getAllCat(self):
        return self.getAll()[_CATEGORIES].as_matrix()

    def getAllX(self):
        return self.getAll()[fields].as_matrix()

    def getAllY(self):
        return self.getAll()[' change'].as_matrix()

if __name__=='__main__':
    ds = Datasets()
    datasets = ds.getByBrands()
    for brand,dataset in datasets.items():
        print(brand,dataset.shape)
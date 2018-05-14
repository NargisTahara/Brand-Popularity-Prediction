import pandas as pd

_FILENAMES = [
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

_FIELDS = [
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
'video_shares',
' change'
]

class Transformer:
    SIZE = 10
    def transform(self, df):
        res = pd.DataFrame()
        res.loc[:,'weekend']=df.loc[:,'weekend']
        res.loc[:,' change']=df.loc[:,' change']
        for field in _FIELDS:
            for i in range(1,len(df)):
                sum = df.loc[max(0,i-self.SIZE):i,field].agg('sum')
                res.loc[i,field] = sum/min(i+1,self.SIZE)
        return res
if __name__=='__main__':
    for filename in _FILENAMES:
        print(filename)
        res = pd.read_csv('./datasets/'+filename)
        t = Transformer()
        df = t.transform(res)
        df.to_csv('./cum10/'+filename)

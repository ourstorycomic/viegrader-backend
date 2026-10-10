import pandas as pd
import io
import re

def auto_clean_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        except:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8-sig')
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError('Chỉ hỗ trợ file .csv, .xls, .xlsx')

    original_cols = list(df.columns)
    lower_cols = [str(c).lower().strip() for c in original_cols]
    text_col = None
    id_col = None
    
    for orig, low in zip(original_cols, lower_cols):
        if not text_col and any(kw in low for kw in ['bai lam', 'text', 'noi dung', 'essay', 'cau tra loi', 'trả lời', 'bài làm']):
            text_col = orig
        if not id_col and any(kw in low for kw in ['id', 'ma', 'mssv', 'stt', 'số thứ tự', 'mã']):
            id_col = orig

    if not text_col:
        text_col = df.astype(str).map(len).mean().idxmax()
        
    if not id_col:
        df['essay_id_auto'] = [f'HS{i+1:04d}' for i in range(len(df))]
        id_col = 'essay_id_auto'

    df = df.dropna(subset=[text_col])
    df = df.rename(columns={text_col: 'text', id_col: 'essay_id'})
    df['text'] = df['text'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    if 'prompt_id' not in df.columns:
        df['prompt_id'] = 'DEFAULT_PROMPT'
        
    return df[['essay_id', 'text', 'prompt_id']]

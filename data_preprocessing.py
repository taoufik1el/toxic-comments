import numpy as np
import pandas as pd

from tqdm.auto import tqdm


RANDOM_SEED = 331
np.random.seed(RANDOM_SEED)


df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test_df = pd.read_csv('../input/jigsaw-toxic-severity-rating/comments_to_score.csv')


df=pd.concat([df[(df.toxic == 1)],
              df[(df.toxic == 1)].sample(20000)]
             )

df=pd.concat([df[(df.toxic==1)&(df.identity_hate==1)],
              df[(df.toxic==1)&(df.threat==0)].sample(2000),
              df[(df.toxic==0)&(df.threat==0)].sample(300),
              df[(df.toxic==0)&(df.threat==1)],
             ])


# from nltk.corpus import stopwords
# stop = stopwords.words('english')

def clean(data, col):
    data[col] = data[col].str.replace('https?://\S+|www\.\S+', ' social medium ', regex=True)

    data[col] = data[col].str.lower()
    data[col] = data[col].str.replace("4", "a")
    data[col] = data[col].str.replace("2", "l")
    data[col] = data[col].str.replace("5", "s")
    data[col] = data[col].str.replace("1", "i")
    data[col] = data[col].str.replace("!", "i")
    data[col] = data[col].str.replace("|", "i", regex=False)
    data[col] = data[col].str.replace("0", "o")
    data[col] = data[col].str.replace("l3", "b")
    data[col] = data[col].str.replace("7", "t")
    data[col] = data[col].str.replace("7", "+")
    data[col] = data[col].str.replace("8", "ate")
    data[col] = data[col].str.replace("3", "e")
    data[col] = data[col].str.replace("9", "g")
    data[col] = data[col].str.replace("6", "g")
    data[col] = data[col].str.replace("@", "a")
    data[col] = data[col].str.replace("$", "s", regex=False)
    data[col] = data[col].str.replace("#ofc", " of fuckin course ")
    data[col] = data[col].str.replace("fggt", " faggot ")
    data[col] = data[col].str.replace("your", " your ")
    data[col] = data[col].str.replace("self", " self ")
    data[col] = data[col].str.replace("cuntbag", " cunt bag ")
    data[col] = data[col].str.replace("fartchina", " fart china ")
    data[col] = data[col].str.replace("youi", " you i ")
    data[col] = data[col].str.replace("cunti", " cunt i ")
    data[col] = data[col].str.replace("sucki", " suck i ")
    data[col] = data[col].str.replace("pagedelete", " page delete ")
    data[col] = data[col].str.replace("cuntsi", " cuntsi ")
    data[col] = data[col].str.replace("i'm", " i am ")
    data[col] = data[col].str.replace("offuck", " of fuck ")
    data[col] = data[col].str.replace("centraliststupid", " central ist stupid ")
    data[col] = data[col].str.replace("hitleri", " hitler i ")
    data[col] = data[col].str.replace("i've", " i have ")
    data[col] = data[col].str.replace("i'll", " sick ")
    data[col] = data[col].str.replace("fuck", " fuck ")
    data[col] = data[col].str.replace("f u c k", " fuck ")
    data[col] = data[col].str.replace("shit", " shit ")
    data[col] = data[col].str.replace("bunksteve", " bunk steve ")
    data[col] = data[col].str.replace('wikipedia', ' social medium ')
    data[col] = data[col].str.replace("faggot", " faggot ")
    data[col] = data[col].str.replace("delanoy", " delanoy ")
    data[col] = data[col].str.replace("jewish", " jewish ")
    data[col] = data[col].str.replace("sexsex", " sex ")
    data[col] = data[col].str.replace("allii", " all ii ")
    data[col] = data[col].str.replace("i'd", " i had ")
    data[col] = data[col].str.replace("'s", " is ")
    data[col] = data[col].str.replace("youbollocks", " you bollocks ")
    data[col] = data[col].str.replace("dick", " dick ")
    data[col] = data[col].str.replace("cuntsi", " cuntsi ")
    data[col] = data[col].str.replace("mothjer", " mother ")
    data[col] = data[col].str.replace("cuntfranks", " cunt ")
    data[col] = data[col].str.replace("ullmann", " jewish ")
    data[col] = data[col].str.replace("mr.", " mister ", regex=False)
    data[col] = data[col].str.replace("aidsaids", " aids ")
    data[col] = data[col].str.replace("njgw", " nigger ")
    data[col] = data[col].str.replace("wiki", " social medium ")
    data[col] = data[col].str.replace("administrator", " admin ")
    data[col] = data[col].str.replace("gamaliel", " jewish ")
    data[col] = data[col].str.replace("rvv", " vanadalism ")
    data[col] = data[col].str.replace("admins", " admin ")
    data[col] = data[col].str.replace("pensnsnniensnsn", " penis ")
    data[col] = data[col].str.replace("pneis", " penis ")
    data[col] = data[col].str.replace("pennnis", " penis ")
    data[col] = data[col].str.replace("pov.", " point of view ", regex=False)
    data[col] = data[col].str.replace("vandalising", " vandalism ")
    data[col] = data[col].str.replace("cock", " dick ")
    data[col] = data[col].str.replace("asshole", " asshole ")
    data[col] = data[col].str.replace("youi", " you ")
    data[col] = data[col].str.replace("afd", " all fucking day ")
    data[col] = data[col].str.replace("sockpuppets", " sockpuppetry ")
    data[col] = data[col].str.replace("iiprick", " iprick ")
    data[col] = data[col].str.replace("penisi", " penis ")
    data[col] = data[col].str.replace("warrior", " warrior ")
    data[col] = data[col].str.replace("loil", " laughing out insanely loud ")
    data[col] = data[col].str.replace("vandalise", " vanadalism ")
    data[col] = data[col].str.replace("helli", " helli ")
    data[col] = data[col].str.replace("lunchablesi", " lunchablesi ")
    data[col] = data[col].str.replace("special", " special ")
    data[col] = data[col].str.replace("ilol", " i lol ")
    data[col] = data[col].str.replace(r'\b[uU]\b', 'you', regex=True)
    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'s", " is ", regex=False)
    data[col] = data[col].str.replace(r"\'ve", " have ", regex=False)
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ", regex=False)
    data[col] = data[col].str.replace(r"\'d", " would ", regex=False)
    data[col] = data[col].str.replace(r"\'ll", " will ", regex=False)
    data[col] = data[col].str.replace(r"\'scuse", " excuse ", regex=False)
    data[col] = data[col].str.replace('\s+', ' ', regex=True)  # will remove more than one whitespace character
    #     text = re.sub(r'\b([^\W\d_]+)(\s+\1)+\b', r'\1', re.sub(r'\W+', ' ', text).strip(), flags=re.I)  # remove repeating words coming immediately one after another
    data[col] = data[col].str.replace(r'(.)\1+', r'\1\1',
                                      regex=True)  # 2 or more characters are replaced by 2 characters
    #     text = re.sub(r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', text, flags = re.I)
    data[col] = data[col].str.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '', regex=True)

    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'ve", " have ", regex=False)
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ", regex=False)
    data[col] = data[col].str.replace(r"i'm", "i am ", regex=False)
    data[col] = data[col].str.replace(r"\'re", " are ", regex=False)
    data[col] = data[col].str.replace(r"\'d", " would ", regex=False)
    data[col] = data[col].str.replace(r"\'ll", " will ", regex=False)
    data[col] = data[col].str.replace(r"\'scuse", " excuse ", regex=False)
    data[col] = data[col].str.replace(r"\'s", " ", regex=False)

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3', regex=True)
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1', regex=True)
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ', regex=True)
    # patterns with repeating characters
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1', regex=True)
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1', regex=True)
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ', regex=True).str.strip()
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ', regex=True).str.strip()
    tqdm.pandas()
    data[col] = data[col].progress_apply(text_cleaning)
    data[col] = data[col].apply(lambda x: x.lower())
    return data


df = clean(df, 'comment_text')

import wordcount
import gensim
from gensim import corpora

if __name__ == "__main__":
    args = wordcount.parse_arguments()
    records = wordcount.load_records(args.file)
    texts = wordcount.preprocess_records(records)\
        .map(lambda record: record[1])\
        .collect()
    print("=====Texts=====")
    for i in range(5):
        print(texts[i])
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    print("=====10 Topics====\n")
    print(ldamodel.print_topics(num_topics=10, num_words=2))
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=9, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    print("=====9 Topics====\n")
    print(ldamodel.print_topics(num_topics=9, num_words=2))
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=8, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    print("=====8 Topics=====\n")
    print(ldamodel.print_topics(num_topics=8, num_words=2))
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=7, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    print("====7 Topics=====\n")
    print(ldamodel.print_topics(num_topics=7, num_words=2))
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=6, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    print("=====6 Topics=====\n")
    print(ldamodel.print_topics(num_topics=6, num_words=2))

"""
=====10 Topics====

[(0, u'0.020*"storm" + 0.015*"campus" + 0.015*"power" + 0.012*"florence" + 0.011*"emergency"'), (1, u'0.014*"kit" + 0.009*"florence" + 0.008*"city" + 0.008*"time" + 0.007*"item"'), (2, u'0.019*"de" + 0.014*"hurricane" + 0.012*"florence" + 0.010*"pollen" + 0.009*"het"'), (3, u'0.047*"hurricane" + 0.046*"florence" + 0.030*"storm" + 0.027*"north" + 0.014*"coast"'), (4, u'0.012*"hurricane" + 0.010*"trump" + 0.010*"link" + 0.010*"florence" + 0.009*"news"'), (5, u'0.029*"storm" + 0.029*"florence" + 0.026*"hurricane" + 0.021*"north" + 0.017*"forecast"'), (6, u'0.015*"share" + 0.015*"news" + 0.014*"flood" + 0.010*"bring" + 0.009*"area"'), (7, u'0.019*"hurricane" + 0.017*"florence" + 0.014*"news" + 0.013*"storm" + 0.011*"emergency"'), (8, u'0.021*"hurricane" + 0.010*"florence" + 0.009*"area" + 0.006*"site" + 0.006*"people"'), (9, u'0.015*"append" + 0.013*"parallax" + 0.007*"earth" + 0.006*"apparent" + 0.005*"change"')]
=====9 Topics====

[(0, u'0.022*"de" + 0.015*"florence" + 0.012*"hurricane" + 0.012*"kit" + 0.011*"het"'), (1, u'0.008*"florence" + 0.007*"trump" + 0.007*"news" + 0.006*"watch" + 0.006*"city"'), (2, u'0.023*"hurricane" + 0.019*"storm" + 0.012*"north" + 0.011*"weather" + 0.011*"florence"'), (3, u'0.051*"hurricane" + 0.050*"florence" + 0.030*"storm" + 0.026*"north" + 0.016*"coast"'), (4, u'0.034*"share" + 0.015*"post" + 0.012*"flood" + 0.011*"link" + 0.010*"north"'), (5, u'0.032*"florence" + 0.026*"storm" + 0.024*"hurricane" + 0.022*"forecast" + 0.021*"north"'), (6, u'0.015*"ash" + 0.011*"coal" + 0.009*"pest" + 0.008*"climate" + 0.008*"nuclear"'), (7, u'0.013*"caption" + 0.013*"hide" + 0.010*"link" + 0.009*"information" + 0.008*"site"'), (8, u'0.022*"storm" + 0.016*"emergency" + 0.015*"hurricane" + 0.011*"florence" + 0.010*"power"')]
=====8 Topics=====

[(0, u'0.021*"florence" + 0.019*"north" + 0.015*"de" + 0.012*"caption" + 0.011*"hide"'), (1, u'0.030*"storm" + 0.021*"hurricane" + 0.016*"florence" + 0.014*"north" + 0.012*"emergency"'), (2, u'0.020*"share" + 0.010*"news" + 0.008*"hurricane" + 0.007*"link" + 0.007*"post"'), (3, u'0.032*"florence" + 0.029*"storm" + 0.029*"hurricane" + 0.019*"north" + 0.015*"coast"'), (4, u'0.017*"news" + 0.017*"emergency" + 0.010*"typhoon" + 0.009*"oil" + 0.008*"plan"'), (5, u'0.010*"florence" + 0.009*"trump" + 0.007*"president" + 0.006*"information" + 0.006*"century"'), (6, u'0.011*"kit" + 0.009*"pollen" + 0.008*"water" + 0.007*"make" + 0.006*"home"'), (7, u'0.053*"hurricane" + 0.050*"florence" + 0.024*"north" + 0.021*"storm" + 0.016*"coast"')]
====7 Topics=====

[(0, u'0.009*"information" + 0.009*"news" + 0.008*"emergency" + 0.007*"disaster" + 0.007*"power"'), (1, u'0.011*"high" + 0.010*"share" + 0.009*"pollen" + 0.009*"kit" + 0.007*"area"'), (2, u'0.033*"storm" + 0.023*"hurricane" + 0.023*"news" + 0.020*"florence" + 0.010*"people"'), (3, u'0.014*"news" + 0.013*"florence" + 0.012*"de" + 0.010*"hurricane" + 0.007*"storm"'), (4, u'0.045*"florence" + 0.042*"hurricane" + 0.027*"north" + 0.027*"storm" + 0.017*"coast"'), (5, u'0.028*"hurricane" + 0.025*"storm" + 0.022*"florence" + 0.014*"emergency" + 0.011*"north"'), (6, u'0.015*"share" + 0.012*"post" + 0.012*"link" + 0.011*"hurricane" + 0.009*"florence"')]
=====6 Topics=====

[(0, u'0.018*"news" + 0.013*"florence" + 0.012*"share" + 0.011*"hurricane" + 0.009*"de"'), (1, u'0.019*"share" + 0.014*"link" + 0.013*"pollen" + 0.012*"post" + 0.010*"twitter"'), (2, u'0.033*"florence" + 0.022*"storm" + 0.022*"hurricane" + 0.021*"north" + 0.020*"forecast"'), (3, u'0.011*"emergency" + 0.011*"hurricane" + 0.010*"power" + 0.009*"storm" + 0.009*"news"'), (4, u'0.033*"storm" + 0.029*"hurricane" + 0.024*"florence" + 0.022*"north" + 0.013*"people"'), (5, u'0.048*"florence" + 0.048*"hurricane" + 0.018*"north" + 0.018*"storm" + 0.011*"coast"')]
"""
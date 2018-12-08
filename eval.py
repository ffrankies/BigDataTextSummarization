from __future__ import unicode_literals
import argparse
from pythonrouge.pythonrouge import Pythonrouge


def rouge_para(predict_path, golden_path):

    # initialize setting of ROUGE, eval ROUGE-1, 2, SU4
    # if summary_file_exis=True, you should specify predict summary(peer_path) and golden summary(model_path) paths
    rouge = Pythonrouge(summary_file_exist=True,
                        peer_path=predict_path, model_path=golden_path,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                        recall_only=True,
                        stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)

    score = rouge.calc_score()
    print(score)


def rouge_sent(predict_path, golden_path):

    import spacy
    nlp = spacy.load('en_core_web_sm')

    with open(predict_path, 'r') as rf:
        predict_summary = rf.read().replace('\n', ' ')
    with open(golden_path, 'r') as rf:
        golden_summary = rf.read().replace('\n', ' ')

    doc = nlp(predict_summary)
    predict_sent_list = [sent.text for sent in doc.sents]

    doc = nlp(golden_summary)
    golden_sent_list = [sent.text for sent in doc.sents]

    max_rouge_1 = -99999
    max_rouge_2 = -99999

    for p_idx, predict_sent in enumerate(predict_sent_list):
        for g_idx, golden_sent in enumerate(golden_sent_list):

            # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
            # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
            # if recall_only=True, you can get recall scores of ROUGE
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=[[predict_sent]], reference=[[[golden_sent]]],
                                n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                                recall_only=True, stemming=True, stopwords=True,
                                word_level=True, length_limit=True, length=50,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            score = rouge.calc_score()

            if score['ROUGE-1'] > max_rouge_1:
                max_rouge_1 = score['ROUGE-1']
                p_idx_rouge_1_save = p_idx
                g_idx_rouge_1_save = g_idx

            if score['ROUGE-2'] > max_rouge_2:
                max_rouge_2 = score['ROUGE-2']
                p_idx_rouge_2_save = p_idx
                g_idx_rouge_2_save = g_idx

    print("Max ROUGE-1 score among sentences: %.5f" % max_rouge_1)
    print("[Predicted Sentence]\n%s" % predict_sent_list[p_idx_rouge_1_save])
    print("[Golden Sentence]\n%s" % golden_sent_list[g_idx_rouge_1_save])

    print()

    print("Max ROUGE-2 score among sentences: %.5f" % max_rouge_2)
    print("[Predicted Sentence]\n%s" % predict_sent_list[p_idx_rouge_2_save])
    print("[Golden Sentence]\n%s" % golden_sent_list[g_idx_rouge_2_save])


def cov_entity(predict_path, golden_path):

    import spacy
    nlp = spacy.load('en_core_web_sm')

    with open(predict_path, 'r') as rf:
        predict_summary = rf.read().replace('\n', ' ')
    with open(golden_path, 'r') as rf:
        golden_summary = rf.read().replace('\n', ' ')

    doc = nlp(predict_summary)
    predict_entity_list = list(set([ent.text for ent in doc.ents]))

    doc = nlp(golden_summary)
    golden_entity_list = list(set([ent.text for ent in doc.ents]))

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    coverage = len(intersection(predict_entity_list, golden_entity_list)) / len(golden_entity_list)

    print('Entity Coverage: %.2f%%' % (coverage * 100))
    print("[Predicted]")
    print(predict_entity_list)
    print("[Golden]")
    print(golden_entity_list)


def main(args):
    
    print("====Rouge Para====")
    rouge_para('./predict/', './golden/')
    print("====Rouge Sent====")
    rouge_sent('./predict/'+args.predict, './golden/'+args.golden)
    print("====Cov Sent====")
    cov_entity('./predict/'+args.predict, './golden/'+args.golden)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Summary Evaluation")
    parser.add_argument('-t', '--type', default='1', type=str, help="type of evaluation: 1-ROUGE_para, 2-ROUGE_sent, 3-Cov_entity")
    parser.add_argument('-g', '--golden', default='./golden/', type=str, help="folder or file path of golden standard summary")
    parser.add_argument('-p', '--predict', default='./predict/', type=str, help="folder or file path of predicted summary")

    args = parser.parse_args()

    main(args)
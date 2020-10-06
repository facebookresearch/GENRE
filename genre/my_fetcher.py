import pickle
from collections import defaultdict
import numpy as np
import time
import sys

# print("preprocessing/util.py print sys.path")
# print(sys.path)
import os
import string


# import preprocessing.config as config
# import model.config as config

# methods below are executed every time because they are fast or because their result
# depend on the args

base_folder = ""

# def load_entities_universe():
#     entities_universe = set()
#     # TODO this path is hardcoded. these mapping files should be transfered in ./data folder
#     with open("/home/other_projects/deep_ed/data/generated/nick/"
#               "wikiid2nnid.txt") as fin:
#         for line in fin:
#             ent_id = line.split('\t')[0]
#             entities_universe.add(ent_id)
#     return entities_universe


# def load_wikiid2nnid(extension_name=None):
#     """returns a map from wiki id to neural network id (for the entity embeddings)"""
#     wikiid2nnid = dict()   # wikiid is string,   nnid is integer
#     with open(config.base_folder+"data/entities/wikiid2nnid/wikiid2nnid.txt") as fin:
#         for line in fin:
#             ent_id, nnid = line.split('\t')
#             wikiid2nnid[ent_id] = int(nnid) - 1  # torch starts from 1 instead of zero
#         assert(wikiid2nnid["1"] == 0)
#         assert(-1 not in wikiid2nnid)
#         wikiid2nnid["<u>"] = 0
#         del wikiid2nnid["1"]
#         #print(len(wikiid2nnid))

#     if extension_name:
#         load_entity_extension(wikiid2nnid, extension_name)
#     return wikiid2nnid


# def load_entity_extension(wikiid2nnid, extension_name):
#     filepath = config.base_folder + "data/entities/" + extension_name + "/wikiid2nnid/additional_wikiids.txt"
#     max_nnid = max(wikiid2nnid.values())
#     assert(len(wikiid2nnid) - 1 == max_nnid)
#     with open(filepath) as fin:
#         line_cnt = 1
#         for line in fin:
#             ent_id = line.strip()
#             if ent_id in wikiid2nnid:   # if extension entities has overlap with the normal entities set
#                 wikiid2nnid[ent_id + "dupl"] = max_nnid + line_cnt    # this vector is duplicate and is never going to be used
#             else:
#                 wikiid2nnid[ent_id] = max_nnid + line_cnt
#             line_cnt += 1
#     print("original entities: ", max_nnid + 1, " extension entities: ", len(wikiid2nnid) - (max_nnid+1))


def reverse_dict(d, unique_values=False):
    new_d = dict()
    for k, v in d.items():
        if unique_values:
            assert v not in new_d
        new_d[v] = k
    return new_d


# def p_e_m_disamb_redirect_wikinameid_maps():
def load_redirections(lowercase=None):
    if lowercase is None:
        lowercase = config.lowercase_maps
    wall_start = time.time()
    redirections = dict()
    with open(
        "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/basic_data/wiki_redirects.txt"
    ) as fin:
        redirections_errors = 0
        for line in fin:
            line = line.rstrip()
            try:
                old_title, new_title = line.split("\t")
                if lowercase:
                    old_title, new_title = old_title.lower(), new_title.lower()
                redirections[old_title] = new_title
            except ValueError:
                redirections_errors += 1

    print("load redirections. wall time:", (time.time() - wall_start) / 60, " minutes")
    print("redirections_errors: ", redirections_errors)
    return redirections


def load_disambiguations():
    wall_start = time.time()
    disambiguations_ids = set()
    # disambiguations_titles = set()
    disambiguations_errors = 0
    with open(
        "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/basic_data/wiki_disambiguation_pages.txt"
    ) as fin:
        for line in fin:
            line = line.rstrip()
            try:
                article_id, title = line.split("\t")
                disambiguations_ids.add(article_id)
                # disambiguations_titles.add(title)
            except ValueError:
                disambiguations_errors += 1
    print(
        "load disambiguations. wall time:", (time.time() - wall_start) / 60, " minutes"
    )
    print("disambiguations_errors: ", disambiguations_errors)
    return disambiguations_ids


def load_disambiguations2():
    wall_start = time.time()
    #     disambiguations_ids = set()
    disambiguations_titles = set()
    disambiguations_errors = 0
    with open(
        "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/basic_data/wiki_disambiguation_pages.txt"
    ) as fin:
        for line in fin:
            line = line.rstrip()
            try:
                article_id, title = line.split("\t")
                #                 disambiguations_ids.add(article_id)
                disambiguations_titles.add(title)
            except ValueError:
                disambiguations_errors += 1
    print(
        "load disambiguations. wall time:", (time.time() - wall_start) / 60, " minutes"
    )
    print("disambiguations_errors: ", disambiguations_errors)
    return disambiguations_titles


def load_persons():
    wiki_name_id_map, _ = load_wiki_name_id_map()
    persons_wikiids = set()
    not_found_cnt = 0
    with open(
        "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/basic_data/persons.txt"
    ) as fin:
        for line in fin:
            line = line.strip()
            if line in wiki_name_id_map:
                persons_wikiids.add(wiki_name_id_map[line])
            else:
                not_found_cnt += 1
                # print("not found:", repr(line))
    print("persons not_found_cnt:", not_found_cnt)
    return persons_wikiids


def load_wiki_name_id_map(lowercase=False, filepath=None):
    wall_start = time.time()
    wiki_name_id_map = dict()
    wiki_id_name_map = dict()
    wiki_name_id_map_errors = 0
    duplicate_names = 0  # different lines in the doc with the same title
    duplicate_ids = 0  # with the same id
    if filepath is None:
        filepath = "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/basic_data/wiki_name_id_map.txt"
    disambiguations_ids = load_disambiguations()
    with open(filepath) as fin:
        for line in fin:
            line = line.rstrip()
            try:
                wiki_title, wiki_id = line.split("\t")
                if wiki_id in disambiguations_ids:
                    continue
                if lowercase:
                    wiki_title = wiki_title.lower()

                if wiki_title in wiki_name_id_map:
                    duplicate_names += 1
                if wiki_id in wiki_id_name_map:
                    duplicate_ids += 1

                wiki_name_id_map[wiki_title] = wiki_id
                wiki_id_name_map[wiki_id] = wiki_title
            except ValueError:
                wiki_name_id_map_errors += 1
    print(
        "load wiki_name_id_map. wall time:", (time.time() - wall_start) / 60, " minutes"
    )
    print("wiki_name_id_map_errors: ", wiki_name_id_map_errors)
    print("duplicate names: ", duplicate_names)
    print("duplicate ids: ", duplicate_ids)
    return wiki_name_id_map, wiki_id_name_map


class FetchCandidateEntities(object):
    """takes as input a string or a list of words and checks if it is inside p_e_m
    if yes it returns the candidate entities otherwise it returns None.
    it also checks if string.lower() inside p_e_m and if string.lower() inside p_e_m_low"""

    def __init__(self, args):
        self.lowercase_spans = args.lowercase_spans
        self.lowercase_p_e_m = args.lowercase_p_e_m
        self.p_e_m, self.p_e_m_low, self.mention_total_freq = custom_p_e_m(
            cand_ent_num=args.cand_ent_num, lowercase_p_e_m=args.lowercase_p_e_m
        )

    def process(self, span):
        """span can be either a string or a list of words"""
        if isinstance(span, list):
            span = " ".join(span)
        title = span.title()
        # 'obama 44th president of united states'.title() # 'Obama 44Th President Of United States'
        title_freq = (
            self.mention_total_freq[title] if title in self.mention_total_freq else 0
        )
        span_freq = (
            self.mention_total_freq[span] if span in self.mention_total_freq else 0
        )

        if title_freq == 0 and span_freq == 0:
            if self.lowercase_spans and span.lower() in self.p_e_m:
                return map(list, zip(*self.p_e_m[span.lower()]))
            elif self.lowercase_p_e_m and span.lower() in self.p_e_m_low:
                return map(list, zip(*self.p_e_m_low[span.lower()]))
            else:
                return None, None

        else:
            if span_freq > title_freq:
                return map(list, zip(*self.p_e_m[span]))
            else:
                return map(list, zip(*self.p_e_m[title]))

                # from [('ent1', 0.4), ('ent2', 0.3), ('ent3', 0.3)] to
                # ('ent1', 'ent2', 'ent3')  and (0.4, 0.3, 0.3)
                # after map we have lists i.e. ['ent1', 'ent2', 'ent3']   , [0.4, 0.3, 0.3]


class FetchFilteredCoreferencedCandEntities(object):
    def __init__(self, args):
        self.args = args
        self.fetchCandidateEntities = FetchCandidateEntities(args)
        self.el_mode = True
        if args.persons_coreference:
            self.persons_wikiids = load_persons()
            self.persons_mentions_seen = list()

    def init_coref(self, el_mode):
        self.persons_mentions_seen = list()
        self.el_mode = el_mode

    def process(self, left, right, chunk_words):
        left_right_words = (
            [
                chunk_words[left - 1] if left - 1 >= 0 else None,
                chunk_words[right] if right <= len(chunk_words) - 1 else None,
            ]
            if self.el_mode
            else None
        )
        span_text = " ".join(chunk_words[left:right])
        cand_ent, scores = self.fetchCandidateEntities.process(span_text)
        if self.args.persons_coreference:
            coreference_supermention = self.find_corefence_person(
                span_text, left_right_words
            )
            if coreference_supermention:
                # print("original text:", chunk_words[max(0, left-4):min(len(chunk_words), right+4)])
                if not self.args.persons_coreference_merge:
                    cand_ent, scores = self.fetchCandidateEntities.process(
                        coreference_supermention
                    )
                else:  # merge with cand_ent and scores
                    cand_ent2, scores2 = self.fetchCandidateEntities.process(
                        coreference_supermention
                    )
                    temp1 = list(zip(scores, cand_ent)) if scores and cand_ent else []
                    temp2 = (
                        list(zip(scores2, cand_ent2)) if scores2 and cand_ent2 else []
                    )
                    temp3 = sorted(temp1 + temp2, reverse=True)
                    scores, cand_ent = map(list, zip(*temp3[: self.args.cand_ent_num]))

        if cand_ent is not None and scores is not None:
            if (
                self.args.persons_coreference
                and not coreference_supermention
                and cand_ent[0] in self.persons_wikiids
                and len(span_text) >= 3
            ):
                if (
                    not self.el_mode
                    or span_text == span_text.title()
                    or span_text == string.capwords(span_text)
                ):
                    self.persons_mentions_seen.append(span_text)
        return cand_ent, scores

    def find_corefence_person(self, span_text, left_right_words):
        """if span_text is substring of another person's mention found before. it should be
        substring of words. so check next and previous characters to be non alphanumeric"""
        if len(span_text) < 3:
            return None
        if left_right_words:  # this check is only for allspans mode not for gmonly.
            if (
                left_right_words[0]
                and left_right_words[0][0].isupper()
                or left_right_words[1]
                and left_right_words[1][0].isupper()
            ):
                # if the left or the right word has uppercased its first letter then do not search for coreference
                # since most likely it is a subspan of a mention.
                # This condition gives no improvement to Gerbil results even a very slight decrease (0.02%)
                return None
        for mention in reversed(self.persons_mentions_seen):
            idx = mention.find(span_text)
            if idx != -1:
                if len(mention) == len(span_text):
                    continue  # they are identical so no point in substituting them
                if idx > 0 and mention[idx - 1].isalpha():
                    continue
                if (
                    idx + len(span_text) < len(mention)
                    and mention[idx + len(span_text)].isalpha()
                ):
                    continue
                # print("persons coreference, before:", span_text, "after:", mention)
                return mention
        return None


class EntityNameIdMap(object):
    def __init__(self):
        pass

    def init_compatible_ent_id(self):
        self.wiki_name_id_map, self.wiki_id_name_map = load_wiki_name_id_map(
            lowercase=False
        )

    def init_gerbil_compatible_ent_id(self):
        self.wiki_name_id_map, self.wiki_id_name_map = load_wiki_name_id_map(
            lowercase=False
        )
        self.redirections = load_redirections(lowercase=False)

    def init_hyperlink2id(self):
        self.wiki_name_id_map, self.wiki_id_name_map = load_wiki_name_id_map(
            lowercase=False
        )
        self.wiki_name_id_map_l, _ = load_wiki_name_id_map(lowercase=True)
        self.redirections = load_redirections(lowercase=False)
        self.disambiguations = load_disambiguations()
        self.hyperlinks_to_dismabiguation_pages = 0

    def hyperlink2id(self, line):
        """gets as input the raw line:
        <a href="political philosophy">\n
        '<a\xa0href="Anarchist\xa0schools\xa0of\xa0thought">\n'
        """
        # line = '<a href="Anarchist schools of thought">\n'
        line = line.rstrip()
        hyperlink_text = line[9:-2]
        # print(repr(hyperlink_text))
        hyperlink_text = hyperlink_text.replace("\xa0", " ").strip()

        for title in [hyperlink_text, hyperlink_text.title()]:
            # look for redirection
            if title in self.redirections:
                title = self.redirections[title]
            if title in self.wiki_name_id_map:
                return self.wiki_name_id_map[title]

        if hyperlink_text.lower() in self.wiki_name_id_map_l:
            return self.wiki_name_id_map_l[hyperlink_text.lower()]
        else:
            return unk_ent_id

    def is_valid_entity_id(self, ent_id):
        return ent_id in self.wiki_id_name_map

    def compatible_ent_id(self, name=None, ent_id=None):
        """takes as input the name and the entity id found in the dataset. If the entity id
        is also in our wiki_name_id_map then this means that this concept-entity also exist in
        out world and with the same id. If the id is not found in our world then we search for
        the name if it is inside the wiki_name_id_map. if yes then we have the same concept
        in our world but with different id so we return and use our own id from now on.
        if neither the id nor the name is in wiki_name_id_map then we return None i.e.
        unknown concept so skip it from the dataset."""
        if ent_id is not None and ent_id in self.wiki_id_name_map:
            return ent_id
        elif name is not None and name in self.wiki_name_id_map:
            return self.wiki_name_id_map[name]
        else:
            return None

    def gerbil_compatible_ent_id(self, uri):
        from urllib.parse import unquote

        title = unquote(uri)
        title = title[len("http://en.wikipedia.org/wiki/") :].replace("_", " ")
        if title in self.redirections:
            title = self.redirections[title]
        if title in self.wiki_name_id_map:
            return self.wiki_name_id_map[title]
        else:
            print(
                "unknown entity. title_searched:",
                repr(title),
                " original uri:",
                repr(uri),
            )
            return None


def custom_p_e_m(cand_ent_num=15, allowed_entities_set=None, lowercase_p_e_m=False):
    """Args:
    cand_ent_num: how many candidate entities to keep for each mention
    allowed_entities_set: restrict the candidate entities to only this set. for example
    the most frequent 1M entities. First this restiction applies and then the cand_ent_num."""
    wall_start = time.time()
    p_e_m = dict()  # for each mention we have a list of tuples (ent_id, score)
    mention_total_freq = dict()  # for each mention of the p_e_m we store the total freq
    # this will help us decide which cand entities to take
    p_e_m_errors = 0
    entityNameIdMap = EntityNameIdMap()
    entityNameIdMap.init_compatible_ent_id()
    incompatible_ent_ids = 0
    with open(
        "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/basic_data/prob_yago_crosswikis_wikipedia_p_e_m.txt"
    ) as fin:
        duplicate_mentions_cnt = 0
        clear_conflict_winner = 0  # both higher absolute frequency and longer cand list
        not_clear_conflict_winner = 0  # higher absolute freq but shorter cand list
        for line in fin:
            line = line.rstrip()
            try:
                temp = line.split("\t")
                mention, entities = temp[0], temp[2:]
                absolute_freq = int(temp[1])
                res = []
                for e in entities:
                    if len(res) >= cand_ent_num:
                        break
                    ent_id, score, _ = map(str.strip, e.split(",", 2))
                    # print(ent_id, score)
                    if not entityNameIdMap.is_valid_entity_id(ent_id):
                        incompatible_ent_ids += 1
                    elif (
                        allowed_entities_set is not None
                        and ent_id not in allowed_entities_set
                    ):
                        pass
                    else:
                        res.append((ent_id, float(score)))
                if res:
                    if mention in p_e_m:
                        duplicate_mentions_cnt += 1
                        # print("duplicate mention: ", mention)
                        if absolute_freq > mention_total_freq[mention]:
                            if len(res) > len(p_e_m[mention]):
                                clear_conflict_winner += 1
                            else:
                                not_clear_conflict_winner += 1
                            p_e_m[mention] = res
                            mention_total_freq[mention] = absolute_freq
                    else:
                        p_e_m[
                            mention
                        ] = res  # for each mention we have a list of tuples (ent_id, score)
                        mention_total_freq[mention] = absolute_freq

            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                p_e_m_errors += 1
                print("error in line: ", repr(line))

    print("duplicate_mentions_cnt: ", duplicate_mentions_cnt)
    print(
        "end of p_e_m reading. wall time:", (time.time() - wall_start) / 60, " minutes"
    )
    print("p_e_m_errors: ", p_e_m_errors)
    print("incompatible_ent_ids: ", incompatible_ent_ids)

    if not lowercase_p_e_m:  # do not build lowercase dictionary
        return p_e_m, None, mention_total_freq

    wall_start = time.time()
    # two different p(e|m) mentions can be the same after lower() so we merge the two candidate
    # entities lists. But the two lists can have the same candidate entity with different score
    # we keep the highest score. For example if "Obama" mention gives 0.9 to entity Obama and
    # OBAMA gives 0.7 then we keep the 0.9 . Also we keep as before only the cand_ent_num entities
    # with the highest score
    p_e_m_lowercased = defaultdict(lambda: defaultdict(int))

    for mention, res in p_e_m.items():
        l_mention = mention.lower()
        # if l_mention != mention and l_mention not in p_e_m:
        #   the same so do nothing      already exist in dictionary
        #   e.g. p(e|m) has Obama and obama. So when i convert Obama to lowercase
        # I find that obama already exist so i will prefer this.
        if l_mention not in p_e_m:
            for r in res:
                ent_id, score = r
                p_e_m_lowercased[l_mention][ent_id] = max(
                    score, p_e_m_lowercased[l_mention][ent_id]
                )

    print(
        "end of p_e_m lowercase. wall time:",
        (time.time() - wall_start) / 60,
        " minutes",
    )

    import operator

    p_e_m_lowercased_trim = dict()
    for mention, ent_score_map in p_e_m_lowercased.items():
        sorted_ = sorted(
            ent_score_map.items(), key=operator.itemgetter(1), reverse=True
        )
        p_e_m_lowercased_trim[mention] = sorted_[:cand_ent_num]

    return p_e_m, p_e_m_lowercased_trim, mention_total_freq


def get_immediate_files(a_dir):
    return [
        name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))
    ]

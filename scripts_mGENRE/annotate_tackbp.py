import pickle

import jsonlines

with jsonlines.open(
    "/checkpoint/fabiopetroni/mGENRE/data/KILT_format/TACKBP2015/train.jsonl"
) as f:
    data = [e for e in f]

with open(
    "/checkpoint/fabiopetroni/mGENRE/wikidata/freebaseID2wikidataID.pkl", "rb"
) as f:
    freebaseID2wikidataID = pickle.load(f)

nil = 0
found = 0
not_found = 0
for d in data:
    d["output"][0]["answer"] = []
    if "NIL" not in d["output"][0]["KB_ID"]:
        id_ = "/" + d["output"][0]["KB_ID"].replace(".", "/")
        if id_ in freebaseID2wikidataID:
            d["output"][0]["answer"] = freebaseID2wikidataID[id_]
            found += 1
        else:
            not_found += 1
    else:
        nil += 1

for d in data:
    if d["output"][0]["answer"] == [None]:
        d["output"][0]["answer"] = []


for d in data:
    if "wikipedia_urls" in d["output"][0] and not d["output"][0]["answer"]:
        l = [e for e in d["output"][0]["wikipedia_urls"] if "en_title" in e]
        if l:
            anchor = (
                unquote(l[0].replace("$00", "%"))
                .replace("/wikipedia/en_title/", "")
                .replace('"', "")
            )
            d["output"][0]["answer"] = list(
                get_wikidata_ids(
                    anchor,
                    "en",
                    lang_title2wikidataID,
                    lang_redirect2title,
                    label_or_alias2wikidataID,
                )[0]
            )

for d in data:
    if "wikipedia_urls" in d["output"][0] and not d["output"][0]["answer"]:
        l = [e for e in d["output"][0]["wikipedia_urls"] if "en_title" in e]
        if l:
            anchor = (
                unquote(l[0].replace("$00", "%"))
                .replace("/wikipedia/en_title/", "")
                .replace('"', "")
            )
            anchor = {
                "J�r�me_Champagne": "Jérôme_Champagne",
                "Fran�ois_De_Keersmaecker": "François_De_Keersmaecker",
                "Citro�n": "Citroën",
                "Gast�n_Pons_Muzzo": "Gastón_Pons_Muzzo",
                "Ilia_Calder�n": "Ilia_Calderón",
                "Jorge_Rodr�guez_(politician)": "Jorge_Rodríguez_(politician)",
                "Mario_Brice�o_Iragorry": "Mario_Briceño_Iragorry",
                "Alexandru_G$0103van": "Alexandru_Găvan",
                "Alo�zio_Mercadante": "Aloízio_Mercadante",
                "Radovan_Krej$010D�$0159": "Radovan_Krejčíř",
                "Rodrigo_Malmierca_D�az": "Rodrigo_Malmierca_Díaz",
                "Universidad_An�huac_M�xico_Norte": "Universidad_Anáhuac_México",
                "Aurelio_Nu�o_Mayer": "Aurelio_Nuño_Mayer",
                "Jos�_Antonio_Fern�ndez": "José_Antonio_Fernández",
                "Luis_D'El�a": "Luis_D%27Elía",
            }.get(anchor, anchor)
            d["output"][0]["answer"] = list(
                get_wikidata_ids(
                    anchor,
                    "en",
                    lang_title2wikidataID,
                    lang_redirect2title,
                    label_or_alias2wikidataID,
                )[0]
            )

for d in data:
    if "wikipedia_urls" in d["output"][0] and not d["output"][0]["answer"]:
        l = [e for e in d["output"][0]["wikipedia_urls"] if "en_title" in e]
        if l:
            anchor = (
                unquote(l[0].replace("$00", "%"))
                .replace("/wikipedia/en_title/", "")
                .replace('"', "")
            )
            print(anchor)

for e in data:
    if "predictions" in e:
        del e["predictions"]


with jsonlines.open(
    "/checkpoint/fabiopetroni/mGENRE/data/KILT_format/TACKBP2015/train.jsonl", "w"
) as f:
    f.write_all(data)

#             outputs = [
#             [[hyp["text"], hyp["logprob"]] for hyp in sent]
#             for sent in outputs
#         ]
#         self.redirections = load_redirections(lowercase=False)


#         def decode(tokens):
#             tokens = "{ ", tokens)
#             tokens = re.sub(r"}.*?", "} ", tokens)
#             tokens = re.sub(r"\].*?", "] ", tokens)
#             tokens = re.sub(r"\[.*?", "[ ", tokens)
#             tokens = re.sub(r"\s{2,}", " ", tokens)
#             return tokens

# inputs = [(
#     " {} ".format(e)
#     .replace(u'\xa0',' ')
#     .replace("{", "(")
#     .replace("}", ")")
#     .replace("[", "(")
#     .replace("]", ")")

# ) for e in inputs]


# outputs = [
#     {"text"}
#     for o in outputs
# ]

# re.sub(r"\{ (.*?) \} \[ (.*?) \]", r"{\1}[\2]", outputs[0][4]["text"]).strip()


#         def decode(tokens):
#             tokens = self.bart.decode(tokens)
#             tokens = re.sub(r"{.*?", "{ ", tokens)
#             tokens = re.sub(r"}.*?", "} ", tokens)
#             tokens = re.sub(r"\].*?", "] ", tokens)
#             tokens = re.sub(r"\[.*?", "[ ", tokens)
#             tokens = re.sub(r"\s{2,}", " ", tokens)
#             return tokens

#         inputs = [(
#             " {} ".format(e)
#             .replace(u'\xa0',' ')
#             .replace("{", "(")
#             .replace("}", ")")
#             .replace("[", "(")
#             .replace("]", ")")

#         ) for e in inputs]


#     def get_entities(self, input_, output_):

#         input_ = input_.replace(u'\xa0',' ') + "  -"
#         output_ = output_.replace("{ ", "{").replace(" } [ ", "}[").replace(" ]", "]") + "  -"

#         entities = []
#         status = "o"
#         i = 0
#         j = 0
#         while j < len(output_) and i < len(input_):

#             if status == "o":
#                 if input_[i] == output_[j] or (
#                     output_[j] in "()" and input_[i] in "[]{}"
#                 ):
#                     i += 1
#                     j += 1
#                 elif output_[j] == " ":
#                     j += 1
#                 elif input_[i] == " ":
#                     i += 1
#                 elif output_[j] == "{":
#                     entities.append([i, 0, ""])
#                     j += 1
#                     status = "m"
#                 else:
#                     raise RuntimeError

#             elif status == "m":
#                 if input_[i] == output_[j]:
#                     i += 1
#                     j += 1
#                     entities[-1][1] += 1
#                 elif output_[j] == " ":
#                     j += 1
#                 elif input_[i] == " ":
#                     i += 1
#                 elif output_[j] == "}":
#                     j += 1
#                     status = "e"
#                 else:
#                     raise RuntimeError

#             elif status == "e":

#                 if output_[j] == "[":
#                     j += 1
#                 elif output_[j] != "]":
#                     entities[-1][2] += output_[j]
#                     j += 1
#                 elif output_[j] == "]":
#                     entities[-1][2] = entities[-1][2].replace(" ", "_")
#                     if len(entities[-1][2]) <= 1:
#                         del entities[-1]
#                     elif entities[-1][2] == "NIL":
#                         del entities[-1]
#                     elif entities[-1][2] in self.redirections:
#                         entities[-1][2] = self.redirections[entities[-1][2]]

#                     status = "o"
#                     j += 1
#                 else:
#                     raise RuntimeError

#         return entities

#     def get_prediction(self, input_, max_length=384):

#         if max_length < 32:
#             return input_

#         i = 0
#         inputs = []
#         for le in [len(e) for e in utils.chunk_it(
#             input_.split(" "), len(input_.split(" ")) // max_length
#         )] if len(input_.split(" ")) > max_length else [len(input_.split(" "))]:
#             inputs.append(" ".join(input_.split(" ")[i:i+le]))
#             i += le

#         outputs = self.get_predictions(inputs)

#         if any(e == [] for e in outputs):
#             print("failed with max_length={}".format(max_length))
#             return self.get_prediction(input_, max_length=max_length // 2)

#         combined_outputs = outputs.pop(0)
#         while len(outputs) > 0:
#             combined_outputs = [
#                 [tp + " " + t, sp + s]
#                 for t, s in outputs.pop(0)
#                 for tp, sp in combined_outputs
#             ]

#         outputs = combined_outputs

#         print("outputs:")
#         pprint(outputs)

#         output_ = re.sub(r"\s{2,}", " ", outputs[0][0])
#         output_ = re.sub(r"\. \. \} \[ (.*?) \]", r". } [ \1 ] .", output_)
#         output_ = re.sub(r"\, \} \[ (.*?) \]", r" } [ \1 ] ,", output_)
#         output_ = re.sub(r"\; \} \[ (.*?) \]", r" } [ \1 ] ;", output_)

#         print("output:", output_)
#         return self.get_entities(input_, output_)

#     def get_markdown(self, sent, spans):
#         text = ""
#         last_end = 0
#         for begin, length, href in spans:
#             text += sent[last_end:begin]
#             text += "[{}](https://en.wikipedia.org/wiki/{})".format(sent[begin:begin + length], href)
#             last_end = begin + length

#         text += sent[last_end:]

#         return Markdown(text)

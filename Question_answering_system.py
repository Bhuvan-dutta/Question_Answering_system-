# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:51:30 2020

@author: bhuva_pxpvpbh
"""

########################Package declaration############################################

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import io
import operator
import spacy

########################Declaring variables############################################ 

nlp = spacy.load("en_core_web_sm")
new_line='\n'
paragraph_separator='. '
lines_separator='\.+\D'
One=1
Zero=0
file_mode='rb'
encoding='utf-8'
exact_line_result=[]
cosine_result=[]
entity_dictionary={}
language='english'
boolean_True=True
Set=set()
sqrt_value=0.5

sw = stopwords.words(language)  

################################Required Input from users##################################

User_Question ="Your Question which user need to ask"
pdf_path='path of the pdf file from which need to aask question'

#######################Creating mask and their entity mappings###################### 

mask_list=["What","Who","When","How many","How much","Why","How"]
mask_direct_dictionary={'What':['PRODUCT','EVENT','LAW','LANGUAGE','CARDINAL','MONEY','PERCENT','WORK_OF_ART'],
                        'Who':['PERSON','NOORP','ORG','EVENT'],
                        'Where':['GPE','LOC','FAC'],
                        'When':['TEMPORAL','DATE','TIME'],
                        'How much':['NUMERIC','PERCENT','CARDINAL','QUANTITY','MONEY'],
                             'How many':['NUMERIC','PERCENT','CARDINAL','QUANTITY','MONEY']}

mask_explanation_dictionary={'Why':['ADP','SCONJ','VERB'],'How':['ADP','VERB']}

#######################Function for extracting raw text data from PDF################

def pdf_to_text_conversion(path):
    
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr,laparams=laparams)
    fp = open(path,file_mode)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    maxpages = Zero
    caching = boolean_True
    pagenos = Set
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,caching=caching,check_extractable=boolean_True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    text = retstr.getvalue()
    retstr.close()
    
    return text

#################stripping the sentences from raw text and splitting paragraph wise##############

Lines=[x.strip() for x in pdf_to_text_conversion(pdf_path).replace(new_line,'').strip().split(paragraph_separator)]

#################Removing blank strings('') from list of sentences###################################

Lines = list(filter(None, Lines))

###########################use gensim or spacy for this function jo thk se ouput de paragraph ka##############

for p in Lines:
    
    X_list = word_tokenize(p)  
    Y_list = word_tokenize(User_Question) 

    context_list =[]
    Question_list =[] 
    c=Zero
    
    #####################remove stop words from string###################################
    
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    ######################form a set containing keywords of both strings#################
    
    rvector = X_set.union(Y_set)
    [context_list.append(One) if word in X_set else context_list.append(Zero) for word in rvector]
    [Question_list.append(One) if word in Y_set else Question_list.append(Zero) for word in rvector]

    #####################finding dot products between normalized vectors##################
    
    c=sum([context_list[i]*Question_list[i] for i in range(len(rvector))])

    if float((sum(context_list)*sum(Question_list))**sqrt_value)!=Zero:
        dot_product_vectors = c / float((sum(context_list)*sum(Question_list))**sqrt_value)
    cosine_result.append(dot_product_vectors)  

max_index, max_value = max(enumerate(cosine_result), key=operator.itemgetter(One))

RelevantContext=Lines[max_index]

print("+++++++++++++++++++++++++++++++++++++")

print("Paragraph in which answer lies:")

print("+++++++++++++++++++++++++++++++++++++")

print(RelevantContext)

print("+++++++++++++++++++++++++++++++++++++")


####################################this function for finding exact line from paragraph######################
doc = nlp(RelevantContext)
entity_explanation_dictionary={}

for ent in doc.ents:
    entity_dictionary[ent.text]=ent.label_ 
for token in doc:
    entity_explanation_dictionary[token.text]=token.pos_

mask = ''.join(list(filter(User_Question.startswith, mask_list)))
if mask in mask_direct_dictionary:
    main_objective_to_find=mask_direct_dictionary.get(mask)
    entity_dictionary=entity_dictionary
elif mask in mask_explanation_dictionary:
    main_objective_to_find=mask_explanation_dictionary.get(mask)
    entity_dictionary=entity_explanation_dictionary
    
#parapgraph_lines=RelevantContext.split('. ')

parapgraph_lines=RelevantContext.splitlines()

for name,entity in entity_dictionary.items():
    if entity in main_objective_to_find:
        Named_entity_list = [key  for (key, value) in entity_dictionary.items() if value == entity]
        for entities in Named_entity_list:
            for lines in parapgraph_lines:
                entity_list = word_tokenize(entities)  
                paragraph_lines_list = word_tokenize(lines) 
    
                ##################sw contains the list of stopwords####################
    
                ent_list =[]
                para_list =[] 
                c=Zero
    
                ####################remove stop words from list#######################
    
                ent_set = {word for word in entity_list if not word in sw}  
                para_set = {word for word in paragraph_lines_list if not word in sw} 
    
                ####################form a set containing keywords of both strings#######
    
                rvector = ent_set.union(para_set)                  
                [ent_list.append(One) if word in ent_set else ent_list.append(Zero) for word in rvector]
                [para_list.append(One) if word in para_set else para_list.append(Zero) for word in rvector]
    
                ####################finding dot products between normalized vectors########
    
                c=sum([ent_list[i]*para_list[i] for i in range(len(rvector))])

                if float((sum(ent_list)*sum(para_list))**sqrt_value)!=Zero:
                    dot_product_vectors =c/float((sum(ent_list)*sum(ent_list))**sqrt_value)
                    
                exact_line_result.append(dot_product_vectors)  

max_index, max_value = max(enumerate(exact_line_result), key=operator.itemgetter(One))

print("+++++++++++++++++++++++++++++++++++++")
print("Here is my exact answer:")
print("+++++++++++++++++++++++++++++++++++++")
print(parapgraph_lines[max_index])
print("+++++++++++++++++++++++++++++++++++++")




from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Distance, VectorParams, 
    SparseVectorParams, SparseIndexParams,
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from langchain_groq import ChatGroq
import json
from google.oauth2.service_account import Credentials
import gspread
from google.oauth2.service_account import Credentials



base_path = "ADD BASE PATH"

files = [
    (base_path + "KB_manual_split_ac.md", "ac_manual", "split ACs"),
    (base_path + "KB_manual_washing_machine.md", "washing_machine_manual", "washing machines"),
    (base_path + "KB_manual_refrigerator.md", "refrigerator_manual", "refrigerators"),
    (base_path + "KB_fault_codes.md", "fault_codes_reference", None),
    (base_path + "KB_safety_bulletin_SB-2026-03.md", "safety_bulletin_SB-2026-03", "washing machines"),
    (base_path + "KB_service_history.csv", "service_history_csv", None)
]

prompt="""You are an appliance classifier for a field-service assistant. Zenith Home Appliances services exactly three product lines. Your only job is to decide which one (if any) the user query refers to.

VALID APPLIANCES (return exactly one of these, or null):
- split ACs       — brand: IntelliCool, models: ZAC-12INV, ZAC-18INV
- washing machines — brand: AquaWash, model: ZWM-70FL
- refrigerators    — brand: FrostGuard, model: ZRF-350DD

MATCH ON:
1. Explicit brand or model names, or close variants and typos of them.
2. Symptom language strongly associated with one appliance:
   - gas smell, hissing, refrigerant leak, compressor, breaker trip, not cooling plus outdoor unit -> split ACs
   - not draining, wont spin, door wont open, vibration, water inlet, pump filter -> washing machines
   - not cooling plus interior light works, frost buildup, water pooling inside, door seal -> refrigerators

RETURN null WHEN:
- The query names a product Zenith does not service, such as microwave, oven, or dishwasher. Do not force-fit it into one of the three.
- The query references two different appliances at once, such as comparing an AC error code to a washing machine error code. Do not guess one, return null and let the retrieval layer search unfiltered.
- There is no appliance-identifying signal at all.

OUTPUT FORMAT — return ONLY a JSON object with a single key named appliance_name, whose value is one of: split ACs, washing machines, refrigerators, or null. No other text, no explanation.

EXAMPLES:
Query: What does error E3 mean on the IntelliCool split AC, and what should the customer do?
Output key appliance_name value: split ACs

Query: A customer smells gas and hears hissing near the outdoor AC unit.
Output key appliance_name value: split ACs

Query: A customer says the washer will not spin.
Output key appliance_name value: washing machines

Query: The FrostGuard fridge is not cooling but the interior light works.
Output key appliance_name value: refrigerators

Query: What is the difference between E5 and F5?
Output key appliance_name value: null

Query: How do I fix the E2 error on my Zenith microwave oven?
Output key appliance_name value: null

DO NOT PRINT ANYTHING ELSE DO NOT WRITE ANYTHING ELSE ONLY IDENTIFY THE APPLIANCE FROM THE USER QUERY AND OUTPUT IN JSON AS REQUIRED
"""


prompt_compile='''You are the answer-generation step of a grounded field-service assistant for Zenith Home Appliances. You help a non-technical Level-1 support agent turn retrieved knowledge-base passages into a correct, safe, cited response for the caller.

You receive:
- USER_QUERY: the caller's problem in plain language.
- CHUNKS: retrieved passages. Each chunk carries metadata: document (file name), appliance, section/heading, and where present error_code. CHUNKS are your ONLY source of truth.

## Core rules
1. Ground every claim in CHUNKS. Never use outside knowledge; never guess or invent facts, codes, quantities, or steps. If a needed fact is not in CHUNKS, you do not have it.
2. Every factual claim must be traceable to a chunk, and you must cite that chunk's document name and its section or error code.
4. Safety first. When a case is safety-critical or you are in any doubt about safety, escalate instead of offering a fix.
5. If no chunk actually addresses the query, abstain — do not stretch an unrelated chunk into an answer.
6. Any query regarding an appliance that is not an AC or a Washing Machine or a refrigerator must be rejected and told so that it is not mentioned 

## Choose exactly one mode

ANSWER — the fix is present in CHUNKS and is safe for a customer / L1 agent to perform.
- What exactly is happening and what is causing it.
- Give step-by-step guidance drawn only from CHUNKS.
- Include any safety precaution the chunk states (e.g. "unplug before opening the pump filter").

ESCALATE — route to a qualified technician. Use this when ANY of the following hold:
- The query contains danger language: gas or refrigerant smell, hissing, burning smell, smoke, electric shock, or water near electricity — escalate even if no chunk matches exactly.
- The matched chunk marks the fault technician-only or says do NOT DIY (e.g. refrigerant leak / E3, compressor overload / E4, any sealed refrigerant-circuit or electrical work).
- Give the immediate safety steps from CHUNKS (e.g. switch off at the breaker, ventilate, unplug) AND state clearly that a certified/qualified technician must handle it. Never provide a DIY refrigerant or electrical repair.
- Escalate if the chunks mention that a technician is needed

ABSTAIN — the answer is not in the knowledge base. Do not fabricate. Two distinct cases:
- out_of_scope: the product is not one the KB covers (only split AC, washing machine, refrigerator). State that the product is out of scope.
- not_in_kb: the appliance is covered but the specific fact is absent from CHUNKS (e.g. a warranty period the docs do not state). State that the answer is not in the knowledge base.

## Conflict resolution
If a safety bulletin and a manual give conflicting instructions for the same fault, follow the bulletin ONLY when its stated scope applies to this unit (matching manufacture date / serial range). When it applies: follow the bulletin, ignore the superseded manual step, and cite the bulletin. If the scope does not apply, follow the manual. If you cannot tell whether the scope applies, escalate.


## Multi-part questions
If the query needs evidence from more than one chunk or source (e.g. a fault code plus service history, or a comparison such as E5 vs F5), combine them and cite each source used.

IF THE ANSWER TO THE QUERY IS NOT IN THE CONTENT JUST SAY SO
THE ANSWER SHOULD CONTAIN THE FIX, THE CAUSE, AND A BIT OF DETAIL REGRADING BOTH. AVOID VERY SHORT ANSWERS AND GIVE DETAIL

## Output
Return ONLY a JSON object — no markdown, no code fences:
{{
  "decision": "answer" | "escalate" | "abstain",
  "abstain_reason": "out_of_scope" | "not_in_kb" | answer | danger,
  "response": "<answer>",
  "citations": ["<document name — section or code>", ...]
}}
Field rules:
- citations reference only documents present in CHUNKS; may be empty when decision is "abstain".
- abstain_reason is null unless decision is "abstain".
- response contains no claim unsupported by CHUNKS. '''




appliance_signals = {
    "split ACs": ["Split AC", "IntelliCool", "ZAC-12INV", "ZAC-18INV", "ZAC"],
    "washing machines": ["Washing machine", "AquaWash", "ZWM-70FL", "ZWM"],
    "refrigerators": ["Refrigerator", "FrostGuard", "ZRF-350DD", "ZRF"],
}


text_chunks=[]
table_chunks=[]
csv_chunks=[]

chunk_size=300
chunk_overlap=60

returned_chunks=6
RFF_chunks=2
cross_encoder_chunks=5

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0,api_key='ADD API KEY',
               model_kwargs={"response_format": {"type": "json_object"}})

splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

embedder= SentenceTransformer("BAAI/bge-base-en-v1.5")
sparse_embedder=SparseTextEmbedding(model_name='Qdrant/bm25')

ranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

client=QdrantClient(":memory:")
client.recreate_collection(collection_name='zenith_kb',
                           vectors_config={'dense':(VectorParams(size=768, distance=Distance.COSINE))},
                           sparse_vectors_config={'sparse':(SparseVectorParams(index=SparseIndexParams(on_disk=False)))})




def appliance_table(heading):
    if not heading:
        return None
    for key,value in appliance_signals.items():
        for i in value:
            if i in heading:
                return key
    return None
    
def splitter_function(chunk_list):
    list_of_list_of_chunks=[]
    for chunk in chunk_list:
        list_of_list_of_chunks.append(splitter.split_text(chunk))
    return list_of_list_of_chunks


def loader_splitter(file_path,file_name,appliance_name):


    global text_chunks, table_chunks
    text=open(file_path,encoding="utf-8").read()
    lines=text.split("\n")

    table_heading=[]
    table_prose=[]
    current_table_heading=''
    current_table_prose=[]

    text_heading=[]
    text_prose=[]
    current_text_heading=''
    current_text_prose=''

    for index,line in enumerate(lines):
        if line.startswith('#') and (index+1 != len(lines)):
            if '|' in lines[index+1]:
                if not current_table_heading == '':
                    table_heading.append(current_table_heading)
                    table_prose.append(current_table_prose)
                current_table_heading=line
                current_table_prose=[]
            else:
                if not current_text_heading == '':
                    text_heading.append(current_text_heading)
                    text_prose.append(current_text_prose)
                current_text_heading=line
                current_text_prose=''
        elif '|' in line:
            current_table_prose.append(line)
        elif not line.startswith('#'):
            current_text_prose=current_text_prose+' '+line

    if not current_text_heading == '':
        text_heading.append(current_text_heading)
        text_prose.append(current_text_prose)
    if not current_table_heading == '':
        table_heading.append(current_table_heading)
        table_prose.append(current_table_prose)

    for index,i in enumerate(table_heading):
        if file_name=='fault_codes_reference':
            table_chunks.append({'content':"\n".join(table_prose[index]),
                                       'type':'table',
                                       'appliance_name':appliance_table(i),
                                       'file_name':file_name,
                                       'section':i})
        else:
           table_chunks.append({'content':"\n".join(table_prose[index]),
                                                  'type':'table',
                                                  'appliance_name':appliance_name,
                                                  'file_name':file_name,
                                                  'section':i}) 

    list_of_list_of_chunks=splitter_function(text_prose)  
    for index,i in enumerate(list_of_list_of_chunks):
        for x in i:
            text_chunks.append({'content':x,
                                                   'type':'text',
                                                   'appliance_name':appliance_name,
                                                   'file_name':file_name,
                                                   'section':text_heading[index]})


def csv_extractor(file_path,file_name):
    df=pd.read_csv(file_path,header=None)
    df = df.fillna("")

    header_row=df.iloc[0]
    csv_content=[]
    global csv_chunks

    for index1,row in df.iterrows():
        if index1>= 1:
            for header,item in zip(header_row,row):
                csv_content.append(str(header)+":"+str(item))

    chunks = [','.join(csv_content[i:i + df.shape[1]]) for i in range(0, len(csv_content), df.shape[1])]


    for i in chunks:
        csv_chunks.append({'content':i,
                      'appliance_name':appliance_table(i),
                      'type':'row',
                      'section':file_name,
                      'file_name':file_name})




for index,i in enumerate(files):
    if (files[index][0]).endswith('.md'):
        loader_splitter(files[index][0],files[index][1],files[index][2])
    elif (files[index][0]).endswith('.csv'):
        csv_extractor(files[index][0],files[index][1])

all_chunks=text_chunks+table_chunks+csv_chunks


#embeddings
contents=[]
for i in all_chunks:
    contents.append(i['content'])

dense_vectors=embedder.encode(contents, normalize_embeddings=True)
sparse_vectors=list(sparse_embedder.embed(contents))
    
points=[]
point_id=0

for index,chunk in enumerate(all_chunks):

    chunk['chunk_id']=index

    dense_vector=dense_vectors[index].tolist()
    sparse_vector=sparse_vectors[index].as_object()

    payload={'content':chunk['content'],
                 'appliance_name':chunk['appliance_name'],
                 'type':chunk['type'],
                 'section':chunk['section'],
                 'file_name':chunk['file_name'],
                 'chunk_id':chunk['chunk_id']}
        
    point=PointStruct(id=point_id, vector={'dense':dense_vector,'sparse':sparse_vector},payload=payload)
        
    points.append(point)
    point_id=point_id+1

client.upsert(collection_name='zenith_kb', points=points)
print("stored", client.count("zenith_kb").count, "points")

print(client.count("zenith_kb").count) 




run=True

def loop():
    global run
    query_text=input('Q: ')

    if query_text=='exit':
        run= False
        return None

    def appliance(heading):
        if not heading:
            return None
    
        full_prompt=prompt+f"user query: {heading}"
        results= llm.invoke(full_prompt).content.strip()
        json_results=json.loads(results)
        return json_results['appliance_name']



    def retrieval(query_text):
        query_appliance=appliance(query_text)
        print(query_appliance)


        dense_query_vector=embedder.encode("Represent this sentence for searching relevant passages: "+query_text,normalize_embeddings=True)
        sparse_query_vector=list(sparse_embedder.query_embed(query_text))[0].as_object()

        #appliance filter
        if query_appliance is not None:
          appliance_filter = Filter(must=[FieldCondition(key='appliance_name', match=MatchValue(value=query_appliance))])
        else:
            appliance_filter = None

        results=[]

        for i in range(len(files)):

            file_filter=Filter(must=[FieldCondition(key='file_name', match=MatchValue(value=files[i][1]))])
            if appliance_filter is not None:
                main_filter=Filter(must=appliance_filter.must + file_filter.must)
            else:
                main_filter=Filter(must=file_filter.must)

            retrieved_results=client.query_points(collection_name='zenith_kb',
                                                prefetch=[Prefetch(query=dense_query_vector, using='dense', limit=returned_chunks, filter= main_filter),
                                                Prefetch(query=sparse_query_vector, using='sparse', limit=returned_chunks, filter=main_filter)],

                                                query=FusionQuery(fusion=Fusion.RRF),
                                                limit=RFF_chunks)
            for i in retrieved_results.points:
                results.append(i)

        return results



    #cross encoder
    query_chunk_pairs=[]
    retrieved_results=retrieval(query_text)

    for chunk in retrieved_results:
        query_chunk_pairs.append((query_text,chunk.payload['content']))

    scores=ranker.predict(query_chunk_pairs)

    scored=list(zip(retrieved_results,scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    #print(scored[:cross_encoder_chunks])
    print(f"\nQuery: {query_text}\n")
    print("="*80)

    for rank, (point, cross_score) in enumerate(scored[:cross_encoder_chunks], start=1):
        payload = point.payload

        print(f"Rank {rank}  |  cross-encoder score: {cross_score:.4f}  |  RRF score: {point.score:.4f}")
        print(f"  chunk_id      : {payload['chunk_id']}")
        print(f"  file_name     : {payload['file_name']}")
        print(f"  appliance_name: {payload['appliance_name']}")
        print(f"  section       : {payload['section']}")
        print(f"  type          : {payload['type']}")
        print(f"  content       : {payload['content'].strip()}")
        print("-"*80)


    def format_chunks_for_llm(scored, top_n=cross_encoder_chunks):
        formatted_chunks = []

        for rank, (point, cross_score) in enumerate(scored[:top_n], start=1):
            payload = point.payload
            chunk_block = f"""[Source {rank}]
    Document: {payload['file_name']}
    Section: {payload['section']}
    Appliance: {payload['appliance_name']}
    Content: {payload['content'].strip()}
    """
            formatted_chunks.append(chunk_block)

        return "\n".join(formatted_chunks)



    prompt1=f'system message:{prompt_compile}, query:{query_text}, content:{format_chunks_for_llm(scored,cross_encoder_chunks)}'
    answer= llm.invoke(prompt1)


    gc=gspread.service_account(filename='ADD CREDENTIAL.JSON FILE PATH')
    sh=gc.open_by_key('SHEET ID')
    worksheet=sh.worksheet('Sheet1')

    answer=json.loads(answer.content)

    new_data=[query_text,answer['decision'],answer['abstain_reason'],answer['response'], " ".join(answer['citations'])]

    worksheet.append_row(new_data)

while run==True:
    loop()
import subprocess 
import json
import streamlit as st
import pandas as pd
import imagehash


def calc_dhash(image):
    the_hash = imagehash.dhash(image, 8)
    bits = ""
    for i, v in enumerate(the_hash.hash.flatten()):
        bits += '1' if v else '0'
        the_hex = str(the_hash)

    return bits, the_hex

activities = ["Table Upload", "List of Users", "Threat Terms", "Reverse Image Hash", "Video Timeline"]
choice = st.sidebar.selectbox("Select Activities", activities)
print(choice)
q_dsl = {"query": {"bool": {"should": [],"minimum_should_match": 1}}}


def match_phrase(key, value):
    return {"match_phrase": {key: value}}


def main():
    if choice == "Table Upload":
        st.write("Video Tutorial")
#         video_file = open('myvideo.mov', 'rb')
#         video_bytes = video_file.read()
#         st.video(video_bytes)

        st.title("Query DSL Generator")
        #need to add storage features
        #need to add max limit of 1024    
        try:
            csv_upload = st.file_uploader("Upload CSV")
            uploaded_file = pd.read_csv(csv_upload)
            st.write("Data Preview")
            st.write(uploaded_file.head())
            columns = [cols for cols in uploaded_file.columns]
            filter_split = columns[0].split(':')
            filter_name = filter_split[0]
            items = uploaded_file[columns[0]].tolist()
            for item in items:
                q_dsl['query']['bool']['should'].append(match_phrase(filter_name, item))
            
            st.write("Query DSL")
            query_to_copy = json.dumps(q_dsl, indent=4)
            st.code(query_to_copy, language='json')
            subprocess.run("pbcopy", universal_newlines=True, input=query_to_copy)
            st.sidebar.write("Result:")
            st.sidebar.success("Success: Your filter pill has been copied to your clipboard!")
        except Exception as e:
            print(e)
    
    elif choice == "List of Users":
        try:
            csv_upload = st.file_uploader("Upload a list of users")
            uploaded_file = pd.read_csv(csv_upload)
            st.write("Data Preview")
            st.write(uploaded_file.head())
            columns = [cols for cols in uploaded_file.columns]
            filter_split = columns[0].split(':')
            filter_name = filter_split[0]
            items = uploaded_file[columns[0]].tolist()
            for item in items:
                q_dsl['query']['bool']['should'].append(match_phrase(filter_name, item))

            st.write("Query DSL")
            query_to_copy = json.dumps(q_dsl, indent=4)
            st.code(query_to_copy, language='json')
            subprocess.run("pbcopy", universal_newlines=True, input=query_to_copy)
            st.sidebar.write("Result:")
            st.sidebar.success("Success: Your filter pill has been copied to your clipboard!")        
        except Exception as e:
            print(e)

    elif choice == "Threat Terms":
        st.write("Threat Term Toggles")
        toggle = st.checkbox('Common English Threat terms')
        csv_upload = st.file_uploader("Upload a list of threat terms")
        try:
            uploaded_file = pd.read_csv(csv_upload)
            st.write("Data Preview")
            st.write(uploaded_file.head())
            length = len(uploaded_file['Threat Terms'])
            threats = "("
            if toggle:
                common_terms = ["behead", "blast", "blaze", "bleed out", "bloodbath", "bloodshed", "blow to hell", "blow up", "bodied", "bomb", "break off", "bullets"]
                for i, threat in enumerate(common_terms, 1):
                    if i < length:
                       threats += f"{threat} OR "
                    else:
                        threats += f"{threat})"
            else:
                for i, threat in enumerate(uploaded_file['Threat Terms'], 1):
                    if i < length:
                        threats += f"{threat} OR "
                    else:
                        threats += f"{threat})"     

            entities = "("
            for i, entity in enumerate(uploaded_file['Entities'], 1):
                if i < length:
                    entities += f"{entity} OR "
                else:
                    entities += f"{entity})" 
        
            direct_query = f"norm.body: ({threats} AND {entities})"
            st.code(direct_query)
        except Exception as e:
            print(e)
    elif choice == "Reverse Image Hash":
        pic_upload = st.file_uploader("Upload an image") 
        import dhash
        from PIL import Image
        from collections import defaultdict
        try:
            obj = Image.open(pic_upload)
            bits, hexed = calc_dhash(obj)
            q_dsl['query']['bool']['should'].append(match_phrase("meta.image_archiver.results.dhash_bits",bits))
            query_to_copy = json.dumps(q_dsl, indent=4)
            st.code(query_to_copy, language='json')
            print(query_to_copy)
        except Exception as e:
            print(e)
    elif choice == "Video Timeline":
        from streamlit_terran_timeline import generate_timeline, terran_timeline
        video = "https://www.youtube.com/watch?v=PDXkB-K_rZw"
        timeline = generate_timeline(video)

        start_time = terran_timeline(timeline)

        st.write(f"User clicked on second {start_time}")

if __name__ == '__main__':
    main()



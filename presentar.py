import streamlit as st
import io

# ------------------------------
#       CONFIGURACIÓN INICIAL
# ------------------------------

st.set_page_config(page_title="MultiTool App", layout="wide", initial_sidebar_state="expanded")

# Inyectamos algo de CSS para mejorar la apariencia
st.markdown(
    """
    <style>
    /* Fondo general */
    .main {
        background-color: #f0f2f6;
    }
    /* Fondo de la barra lateral */
    .css-1d391kg { 
        background-color: #e0e6ed;
    }
    /* Estilo de los títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #333333;
    }
    /* Botones más destacados */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------
#           CONFIGURACIÓN
# -------------------------------------

# Diccionarios para modelos, idiomas y traducción
QA_MODELS = {
    "Español": "mrm8488/bert-spanish-cased-finetuned-squad-v1.1-es",
    "Inglés": "distilbert-base-cased-distilled-squad"
}

ASR_LANGUAGE_CODES = {
    "Español": "es-ES",
    "Inglés": "en-US"
}

TRANSLATOR_LANG_CODES = {
    "Español": "es",
    "Inglés": "en",
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it"
}

# -------------------------------------
#          FUNCIONES Y PIPELINES
# -------------------------------------

from googletrans import Translator
import speech_recognition as sr
from transformers import pipeline
from crossref.restful import Works

@st.cache_resource
def get_qa_pipeline(language_choice):
    """Devuelve un pipeline de Q&A según el idioma seleccionado."""
    model_name = QA_MODELS[language_choice]
    return pipeline("question-answering", model=model_name)

@st.cache_resource
def get_summarizer():
    """Carga un modelo de summarization de HuggingFace."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

def translate_text(original_text, src_lang, dest_lang):
    """Traduce el texto usando googletrans."""
    translator = Translator()
    translation = translator.translate(original_text, src=src_lang, dest=dest_lang)
    return translation.text

def record_and_transcribe(duration, language_code):
    """
    Graba audio durante 'duration' segundos y lo transcribe usando
    SpeechRecognition con el idioma 'language_code' (ej. 'es-ES').
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(f"Grabando durante {duration} segundos. ¡Habla ahora en {language_code}!")
        audio_data = recognizer.record(source, duration=duration)
    st.write("Transcribiendo...")
    try:
        text = recognizer.recognize_google(audio_data, language=language_code)
        return text
    except sr.UnknownValueError:
        return "No se pudo entender el audio."
    except sr.RequestError as e:
        return f"Error en la solicitud de transcripción: {e}"

def search_papers(query=None, author=None, from_year=None, until_year=None, max_results=5):
    """
    Busca artículos en Crossref con filtros de query, autor y año.
    """
    w = Works()
    if query and author:
        results_generator = w.query(query, author=author)
    elif query:
        results_generator = w.query(query)
    elif author:
        results_generator = w.query(author=author)
    else:
        return []
    
    # Filtros por año
    filter_dict = {}
    if from_year:
        filter_dict["from-pub-date"] = f"{from_year}-01-01"
    if until_year:
        filter_dict["until-pub-date"] = f"{until_year}-12-31"
    if filter_dict:
        results_generator = results_generator.filter(**filter_dict)
    
    results_list = []
    count = 0
    for item in results_generator:
        results_list.append(item)
        count += 1
        if count == max_results:
            break
    return results_list

# -------------------------------------
#         BARRA LATERAL: AYUDA
# -------------------------------------

with st.sidebar.expander("Acerca de / Ayuda", expanded=True):
    st.markdown(
        """
        **Bienvenido a la MultiTool App**

        Esta aplicación integra tres funcionalidades:

        1. **Traductor de Documentos**  
           - Sube un archivo TXT y tradúcelo a otro idioma.
        2. **Grabación + Q&A Local**  
           - Graba tu voz, transcribe el audio y haz preguntas sobre el contenido utilizando un modelo de Preguntas y Respuestas.
        3. **Buscador de Papers**  
           - Busca artículos académicos a través de Crossref, filtra por autor y año, y genera un resumen del abstract (si está disponible).

        **Instrucciones generales:**  
        - Selecciona el modo deseado desde el menú lateral.
        - Sigue las instrucciones específicas en cada sección.
        - Utiliza los botones para ejecutar las acciones y observa los resultados en la pantalla principal.

        ¡Explora y disfruta de la aplicación!
        """
    )

# -------------------------------------
#          SELECCIÓN DE MODO
# -------------------------------------

st.sidebar.title("Modos de la Aplicación")
mode = st.sidebar.radio(
    "Elige el modo:",
    ("Traductor de Documentos", "Grabación + Q&A Local", "Buscador de Papers")
)

# -------------------------------------
#           INTERFAZ PRINCIPAL
# -------------------------------------

if mode == "Traductor de Documentos":
    st.title("📄 Traductor de Documentos TXT")
    st.subheader("Selecciona los idiomas de origen y destino")

    src_language_label = st.selectbox("Idioma original", list(TRANSLATOR_LANG_CODES.keys()), index=0)
    dest_language_label = st.selectbox("Idioma destino", list(TRANSLATOR_LANG_CODES.keys()), index=1)

    uploaded_file = st.file_uploader("Sube un archivo TXT", type=["txt"])
    if uploaded_file is not None:
        st.info("Archivo cargado con éxito. Procesando traducción...")
        original_text = uploaded_file.getvalue().decode("utf-8")
        translated_text = translate_text(
            original_text,
            src_lang=TRANSLATOR_LANG_CODES[src_language_label],
            dest_lang=TRANSLATOR_LANG_CODES[dest_language_label]
        )
        output = io.StringIO()
        output.write(f"Texto Original ({src_language_label}):\n")
        output.write(original_text + "\n\n")
        output.write(f"Texto Traducido ({dest_language_label}):\n")
        output.write(translated_text)
        
        st.download_button(
            label="Descargar documento traducido",
            data=output.getvalue(),
            file_name="documento_traducido.txt",
            mime="text/plain"
        )

elif mode == "Grabación + Q&A Local":
    st.title("🎙️ Grabador, Transcriptor y Q&A Offline")
    st.subheader("Selecciona el idioma para grabar y realizar preguntas")
    language_choice = st.selectbox("Idioma:", ["Español", "Inglés"], index=0)
    
    if st.button("Grabar y Transcribir 20 segundos"):
        with st.spinner("Grabando y transcribiendo..."):
            transcription = record_and_transcribe(
                duration=20,
                language_code=ASR_LANGUAGE_CODES[language_choice]
            )
        st.success("Transcripción completada")
        st.session_state.transcription = transcription
        st.text_area("Transcripción", transcription, height=150)
    
    if "transcription" in st.session_state:
        st.subheader("Realiza tu pregunta")
        question = st.text_input("Escribe tu pregunta")
        if st.button("Enviar pregunta"):
            with st.spinner("Procesando tu pregunta..."):
                context = st.session_state.transcription
                if not context.strip():
                    st.warning("La transcripción está vacía o no es válida.")
                else:
                    qa_pipe = get_qa_pipeline(language_choice)
                    try:
                        answer = qa_pipe(question=question, context=context)
                        st.success("Respuesta obtenida:")
                        st.write("**Respuesta:**", answer["answer"])
                    except Exception as e:
                        st.error(f"Error al procesar la pregunta: {e}")

elif mode == "Buscador de Papers":
    st.title("🔍 Buscador de Papers con Resumen")
    
    query = st.text_input("Término de búsqueda (opcional)", "")
    author = st.text_input("Autor (opcional)", "")
    
    col1, col2 = st.columns(2)
    with col1:
        from_year = st.number_input("Desde el año (opcional)", min_value=1900, max_value=2100, step=1, value=2018)
    with col2:
        until_year = st.number_input("Hasta el año (opcional)", min_value=1900, max_value=2100, step=1, value=2023)
    
    max_results = st.slider("Número máximo de resultados", 1, 20, 5)
    
    if "results" not in st.session_state:
        st.session_state.results = []
    if "searched" not in st.session_state:
        st.session_state.searched = False
    
    if st.button("Buscar"):
        results = search_papers(
            query=query.strip() if query.strip() else None,
            author=author.strip() if author.strip() else None,
            from_year=from_year,
            until_year=until_year,
            max_results=max_results
        )
        st.session_state.results = results
        st.session_state.searched = True
    
    if st.session_state.searched:
        results = st.session_state.results
        if results:
            st.success(f"Se encontraron {len(results)} resultados.")
            summarizer = get_summarizer()
            
            for i, item in enumerate(results):
                title = item.get("title", ["Título desconocido"])[0]
                authors = item.get("author", [])
                if authors:
                    authors_str = ", ".join(
                        f"{a.get('given', '')} {a.get('family', '')}" for a in authors
                    )
                else:
                    authors_str = "Autores desconocidos"
                
                year = None
                if "published-print" in item:
                    year = item["published-print"]["date-parts"][0][0]
                elif "published-online" in item:
                    year = item["published-online"]["date-parts"][0][0]
                if not year:
                    year = "Año desconocido"
                
                url = item.get("URL", None)
                
                st.markdown(f"### {title}")
                st.markdown(f"**Autores:** {authors_str}")
                st.markdown(f"**Año:** {year}")
                if url:
                    st.markdown(f"[Enlace al paper]({url})")
                
                if st.button("Generar resumen", key=f"resumen_{i}"):
                    abstract = item.get("abstract", None)
                    if abstract:
                        summary = summarizer(abstract, max_length=130, min_length=30, do_sample=False)
                        st.write("**Resumen:**", summary[0]["summary_text"])
                    else:
                        st.warning("No hay abstract disponible para este paper.")
                st.write("---")
        else:
            st.warning("No se encontraron resultados o la búsqueda falló.")

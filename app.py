import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from herramientas import crear_herramientas

# Inicia la aplicación
st.set_page_config(page_title="Asistente de Análisis de Datos con IA", layout="centered")
st.title("Asistente de Análisis de Datos con IA")

# Descripción de la herramienta
st.info("""
Este asistente utiliza un agente, creado con Langchain, para ayudarte a explorar, analizar y visualizar datos de forma interactiva. Basta con subir un archivo CSV y podrás:

**Generar reportes automáticos**:
* **"Reporte de información general"**: presenta la dimensión del Dataframe, nombres y tipos de las columnas, conteo de datos nulos y duplicados, además de sugerencias de tratamientos y análisis adicionales.
* **"Reporte de estadísticas descriptivas"**: muestra valores como media, mediana, desviación estándar, mínimo y máximo; identifica posibles outliers y sugiere próximos pasos con base en los patrones detectados.

**"Hacer preguntas simples sobre los datos"**: como "¿cuál es el promedio de la columna X?", "¿Cuántos registros existen para cada categoría de la columna Y?".

**"Crear gráficos automáticamente"** a partir de preguntas en lenguaje natural.

Ideal para analistas, científicos de datos y equipos que buscan agilidad e insights rápidos con apoyo de IA.
""")

# upload de CSV
st.markdown("### 📂 Realiza la carga de tu archivo CSV")
archivo_cargado = st.file_uploader("Selecciona un archivo CSV", type="csv", label_visibility="collapsed")

if archivo_cargado:
    df = pd.read_csv(archivo_cargado)
    st.success("¡Archivo cargado exitosamente!")
    st.markdown("###  Primeras filas de tu conjunto de datos")
    st.dataframe(df.head())
    
if archivo_cargado is None:
    st.info("Sube un archivo CSV para comenzar.")
    st.stop()

# LLM
API_KEY_GROQ = os.getenv("API_KEY_GROQ")
MODEL_NAME_GROQ = os.getenv("MODEL_NAME_GROQ")

llm = ChatGroq(
    api_key=API_KEY_GROQ,
    model_name=MODEL_NAME_GROQ,
    temperature=0
    )

# Herramentas
tools = crear_herramientas(df)

# Prompt react
df_head = df.head().to_markdown()

prompt_react_es = PromptTemplate(
    input_variables=["input","agent_scratchpad","tools","tool_names"],
    partial_variables={"df_head": df_head},
    template="""
Eres un asistente que responde en castellano.

Tienes acceso a un dataframe pandas llamado `df`.
Aquí están las primeras filas del dataframe, obtenidas usando `df.head().to_markdown()`:
{df_head}

Responde a las siguientes preguntas de la mejor manera posible.
Para este fin, tienes acceso a las siguientes herramientas:
{tools}

Usa el siguiente formato:
Question: la pregunta de entrada que debes responder
Thought: Debes siempre pensar en lo que debes hacer
Action: la acción que será ejecutada, debe ser una de las [{tool_names}]
Action Input: la entrada para la acción
Observation: el resultado de la acción
... (este Thought/Action/Action Input/Observation puede repetirse N veces)
Thought: Ahora sé la respuesta final
Final Answer: Si la 'Question' contiene las frases 'reporte', 'informe' o 'estadística' y la 'Observation' 
contiene el resultado completo de un reporte/informe, devuelve el contenido de la última 'Observation' directamente.
En cualquier otro caso, responde la pregunta del usuario de forma clara y directa, utilizando la información de la última 
'Observation' o de todas las 'Observations' relevantes para la pregunta, adaptando el formato de la respuesta a la 
solicitud específica del usuario (ej. resumen, explicación, etc.).

Comienzal
Question: {input}
Thought: {agent_scratchpad}
"""
)

# Agente
agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_es)
orquestador = AgentExecutor(agent=agente,
                            tools=tools,
                            verbose=True,
                            handle_parsing_errors=True)

# ACCIONES RÁPIDAS
st.markdown("### ⚡ Acciones rápidas")

# Reporte de Informaciones Generales
if st.button("Reporte de Informaciones Generales", key="boton_reporte_general"):
    with st.spinner("Generando Reporte 🕵️‍♂️..."):
        respuesta = orquestador.invoke({"input": "Quero um relatório com informações sobre os dados"})
        st.session_state['reporte_general'] = respuesta["output"]

# Exhibe el reporte con botón de descarga
if 'reporte_general' in st.session_state:
    with st.expander("Resultado: Reporte de Informaciones Generales"):
        st.markdown(st.session_state['reporte_general'])
    
    st.download_button(
        label="Descargar Reporte",
        data=st.session_state['reporte_general'],
        file_name="reporte_informaciones_generales.md",
        mime="text/markdown"
    )

# Reporte de estadísticas descriptivas
if st.button("Reporte de estadísticas descriptivas", key="boton_reporte_estadisticas"):
    with st.spinner("Generando Reporte 🕵️‍♂️..."):
        respuesta = orquestador.invoke({"input": "Quiero un reporte de estatísticas descritivas"})
        st.session_state['reporte_estadisticas'] = respuesta["output"]

# Exhibe el reporte almacenado con opción de descarga
if 'reporte_estadisticas' in st.session_state:
    with st.expander("Resultado: Reporte de estadísticas descriptivas"):
        st.markdown(st.session_state['reporte_estadisticas'])

    st.download_button(
        label="Descargar Reporte",
        data=st.session_state['reporte_estadisticas'],
        file_name="reporte_estadisticas_descriptivas.md",
        mime="text/markdown"
    )
    
# PREGUNTA SOBRE LOS DATOS
st.markdown("### 🔎 Preguntas sobre los datos")
pregunta_sobre_datos = st.text_input("Realiza una pregunta sobre los datos (ej: '¿Cuál es el promedio de tiempo de entrega?')")
if st.button("Responder pregunta", key="responder_pregunta_datos"):
    with st.spinner("Analizando los datos 🕵️‍♂️..."):
        respuesta = orquestador.invoke({"input": pregunta_sobre_datos})
        st.markdown(respuesta["output"])

# GENERACIÓN DE GRÁFICOS
st.markdown("### 📊 Crear gráfico con base en una pregunta")
pregunta_grafico = st.text_input("Qué deseas visualizar? (ej: 'Genera un gráfico del promedio de tiempo de entrega por clima.')")
if st.button("Generar gráfico", key="generar_grafico"):
    with st.spinner("Generando el gráfico 🎨..."):
        orquestador.invoke({"input": pregunta_grafico})
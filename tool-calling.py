import pandas as pd
import os
from langchain_openai import ChatOpenAI
from pandasql import sqldf
from langchain_core.tools import tool
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage, ToolMessage

os.environ["OPENAI_API_KEY"] = 

colunas = ['DATA (YYYY-MM-DD)', 'Hora UTC', 'TEMPERATURA DO PONTO DE ORVALHO (°C)']

@tool
def maxi_tool(data:str, col:str)->str:
    """imports an array calles data, reads the column 'col' and returns its highest value"""
    df = pd.read_csv('{}.csv'.format(data), usecols=colunas)
    df.columns = ['DATA', 'HORA', 'TEMP']
    x = df["{}".format(col)].max()
    return x
@tool
def min_tool(data:str, col:str)->str:
    """imports an array calles data, reads the column 'col' and returns its lowest value"""
    df = pd.read_csv('{}.csv'.format(data), usecols=colunas)
    df.columns = ['DATA', 'HORA', 'TEMP']
    x = df["{}".format(col)].min()
    return x
tools = [maxi_tool, min_tool]
      
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
query = "What is the highest value of the column 'TEMP' from the dataset  'weather_2021'? "
    
print(llm_with_tools.invoke(query))
messages = [HumanMessage(query)]

llm_output = llm_with_tools.invoke(messages)
messages.append(llm_output)

tool_map = {"maxi_tool":maxi_tool, "min_tool":min_tool}

for tool_call in llm_output.tool_calls:
    tool = tool_map[tool_call['name'].lower()]
    tool_output = tool.invoke(tool_call['args'])
    messages.append(ToolMessage(tool_output,tool_call_id=tool_call['id']))
print(llm_with_tools.invoke(messages).content)



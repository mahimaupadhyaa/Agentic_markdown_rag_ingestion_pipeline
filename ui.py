import gradio as gr
import re
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from typing import Any, Dict, List, Tuple


def preprocess_content(content):
    """Replace XML-style tags with Markdown headings, handling nested tags"""
    if not isinstance(content, str):
        return content
    
    tag_positions = []
    for match in re.finditer(r'</?(\w+)>', content):
        is_closing = match.group(0).startswith('</')
        tag_name = match.group(1)
        tag_positions.append((match.start(), match.end(), tag_name, is_closing))
    
    if not tag_positions:
        return content
    
    tag_positions.sort()
    
    result = []
    last_pos = 0
    nesting_level = 0
    tag_stack = []
    
    for start, end, tag_name, is_closing in tag_positions:
        result.append(content[last_pos:start])
        
        if is_closing:
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
                nesting_level = max(0, nesting_level - 1)
        else:
            heading_level = min(6, 2 + nesting_level)
            result.append('#' * heading_level + ' ' + tag_name.capitalize())
            tag_stack.append(tag_name)
            nesting_level += 1
        
        last_pos = end
    
    result.append(content[last_pos:])
    
    return ''.join(result)


def get_messages_from_langgraph_state(state):
    """Extract ChatMessage objects from agent's state"""
    try:
        messages = state.values['messages']
    except (KeyError, AttributeError):
        messages = []
    
    return messages


def convert_messages_from_langchain_to_gradio(messages):
    """Transform LangChain messages into Gradio ChatMessage objects"""
    
    i = 0
    while i < len(messages):
        message = messages[i]
        
        if isinstance(message, SystemMessage):
            i += 1
            continue
            
        elif isinstance(message, HumanMessage):
            processed_content = preprocess_content(message.content)
            yield gr.ChatMessage(role="user", content=processed_content)
            i += 1
            
        elif isinstance(message, AIMessage):
            ai_content = preprocess_content(message.content)
            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
            
            yield gr.ChatMessage(role="assistant", content=ai_content)
            
            if has_tool_calls and i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage):
                tool_message = messages[i + 1]
                
                for tool_call in message.tool_calls:
                    if tool_call['id'] == tool_message.tool_call_id:
                        
                        parent_id = f"call_{tool_call['id']}"
                        
                        if 'args' in tool_call and isinstance(tool_call['args'], dict):
                            content = str(tool_call['args'].get("answer", str(tool_call['args'])))
                        else:
                            content = preprocess_content(tool_message.content)
                        
                        used_code = tool_call['name'] == "python_interpreter" if 'name' in tool_call else False
                        if used_code:
                            import re
                            content = re.sub(r"```.*?\n", "", content)
                            content = re.sub(r"\s*<end_code>\s*", "", content)
                            content = content.strip()
                            if not content.startswith("```python"):
                                content = f"```python\n{content}\n```"
                        
                        parent_message_tool = gr.ChatMessage(
                            role="assistant",
                            content=content,
                            metadata={
                                "title": f"🛠️ Used tool {tool_call['name']}",
                                "id": parent_id,
                                "status": "pending",
                            }
                        )
                        yield parent_message_tool
                        
                        if hasattr(tool_message, 'observations') and tool_message.observations:
                            log_content = str(tool_message.observations).strip()
                            if log_content:
                                import re
                                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                                yield gr.ChatMessage(
                                    role="assistant",
                                    content=f"```bash\n{log_content}\n```",
                                    metadata={
                                        "title": "📝 Execution Logs",
                                        "parent_id": parent_id,
                                        "status": "done"
                                    }
                                )
                        
                        if hasattr(tool_message, 'error') and tool_message.error is not None:
                            yield gr.ChatMessage(
                                role="assistant",
                                content=str(tool_message.error),
                                metadata={
                                    "title": "💥 Error",
                                    "parent_id": parent_id,
                                    "status": "done"
                                }
                            )
                        
                        parent_message_tool.metadata["status"] = "done"
                        
                        step_footnote = ""
                        if hasattr(message, 'input_token_count') and hasattr(message, 'output_token_count'):
                            token_str = f" | Input-tokens:{message.input_token_count:,} | Output-tokens:{message.output_token_count:,}"
                            step_footnote += token_str
                        
                        if hasattr(message, 'duration'):
                            step_duration = f" | Duration: {round(float(message.duration), 2)}" if message.duration else None
                            if step_duration:
                                step_footnote += step_duration
                        
                        if step_footnote:
                            step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
                            yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
                            yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})
                        
                        break
                
                i += 2
            else:
                i += 1
        
        elif isinstance(message, ToolMessage):
            processed_content = preprocess_content(message.content)
            yield gr.ChatMessage(
                role="assistant",
                content=processed_content,
                metadata={
                    "title": f"🛠️ Tool Result: {message.name}",
                    "status": "done"
                }
            )
            i += 1
        
        else:
            i += 1
    
    return


class GradioUI:

    def __init__(self, agent: Any, config: Dict[str, Any]) -> None:

        print("Initializing Gradio UI")

        self.agent = agent
        self.config = config
        self.responses = []

        self.app = gr.ChatInterface(
            self.interact_with_agent, 
            type="messages"
        )

    def interact_with_agent(self, user: str, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:

        print(f"User message received: {user}")

        msg = get_messages_from_langgraph_state(self.agent.get_state(self.config))
        before = list(convert_messages_from_langchain_to_gradio(msg))

        self.responses = self.agent.invoke(
            {"messages": [{"role": "user", "content": user}]},
            config=self.config
        )

        msg = get_messages_from_langgraph_state(self.agent.get_state(self.config))
        after = list(convert_messages_from_langchain_to_gradio(msg))

        new = after[len(before)+1:]

        print("Agent response generated")

        return new

    def launch(self, **kwargs) -> None:

        print("Launching Gradio interface")

        self.app.launch(**kwargs)


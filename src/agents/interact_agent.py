from ctypes import Union
import mimetypes
from typing import Callable, Generic
from jsonschema import SchemaError
from langchain.agents import create_agent
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import sys
from marshmallow import Schema
from valyu.tools import ValyuSearchTool
from react_agent.holistic_ai_bedrock import HolisticAIBedrockChat, get_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
import importlib.resources
from better_profanity import profanity
file_content = importlib.resources.read_text('better_profanity', 'profanity_wordlist.txt')
profanity_word = file_content.splitlines()

from langchain.chat_models import init_chat_model




@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("📄 Loaded configuration from .env file")
else:
    print("⚠️  No .env file found - using environment variables or hardcoded keys")

# ============================================
# Verify API keys are set
# ============================================
print("\n🔑 API Key Status:")
if os.getenv('HOLISTIC_AI_TEAM_ID') and os.getenv('HOLISTIC_AI_API_TOKEN'):
    print("  ✅ Holistic AI Bedrock credentials loaded (will use Bedrock)")
elif os.getenv('OPENAI_API_KEY'):
    print("  ⚠️  OpenAI API key loaded (Bedrock credentials not set)")
    print("     💡 Tip: Set HOLISTIC_AI_TEAM_ID and HOLISTIC_AI_API_TOKEN to use Bedrock (recommended)")
else:
    print("  ⚠️  No API keys found")
    print("     Set Holistic AI Bedrock credentials (recommended) or OpenAI key")

if os.getenv('VALYU_API_KEY'):
    key_preview = os.getenv('VALYU_API_KEY')[:10] + "..."
    print(f"  ✅ Valyu API key loaded: {key_preview}")
else:
    print("  ⚠️  Valyu API key not found - search tool will not work")

print("\n📁 Working directory:", Path.cwd())



os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = "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"

llm = init_chat_model("gpt-4.1")

# ============================================
# Import Holistic AI Bedrock helper function
# ============================================
# Import from core module (recommended)
import sys
try:
    sys.path.insert(0, '../core')
    from react_agent.holistic_ai_bedrock import HolisticAIBedrockChat, get_chat_model
    print("\n✅ Holistic AI Bedrock helper function loaded")
except ImportError:
    print("\n⚠️  Could not import from core - will use OpenAI only")
    print("   Make sure core/react_agent/holistic_ai_bedrock.py exists")
checkpointer = InMemorySaver()

#llm = get_chat_model("us.anthropic.claude-3-5-sonnet-20241022-v2:0") 
SYSTEM_PROMPT='''
You are a dual-purpose AI assistant for an image generation pipeline. You have two sequential tasks:
1.  **Content Safety Sentry**: First, you must review the user's input for safety.
2.  **Expert Prompt Engineer**: If—and only if—the input is safe, you must expand the user's vague idea into a highly detailed, structured JSON prompt for Stable Diffusion.

---

### TASK 1: CONTENT SAFETY REVIEW

First, you MUST review the <user_input> against the following Safety Policy Categories.

# Safety Policy Categories
1.  **[Violence & Gore]**: Descriptions of extreme violence, war, torture, gore, weapons (especially aimed at people or animals), combat, or severe physical injury.
2.  **[Explicit & Sexual Content]**: Descriptions of nudity (especially in a suggestive context), sexual acts, sexual fetishes, or overly sexualized features.
3.  **[Hate Speech & Discrimination]**: Descriptions of discriminatory or insulting symbols, stereotypes, or scenarios targeting specific groups (based on race, religion, gender, etc.).
4.  **[Illegal Activities & Regulated Goods]**: Descriptions of drug use/manufacturing, gambling, terrorism, theft, or promoting illegal acts.
5.  **[Self-Harm]**: Descriptions or glorification of suicide, self-injury, or severe mental distress.

---

### TASK 2: EXPERT PROMPT ENGINEERING

**If, and only if, the <user_input> is SAFE**, you must perform this task.
Your job is to take the user's simple, vague request and creatively expand it into the detailed JSON structure provided below. You must infer and add details to create a high-quality, artistic image.

# Target JSON Schema & Instructions

You must fill out the following structure. 
Be creative and add common "magic words" (like "masterpiece", "trending on artstation") to enhance the prompt.
### 1."safe": Boolean,whether it is safe
### 2.data_uri = "data:{mime_type};base64,{base64_data}",an image sent by user, you need to contain its{mime_type}and{base64_data},if there is no image in user message kept it an empty string.
### 3. `prompt` Object
* **subject**: Define the main focus. Infer `details`, `actions`, and `physical_attributes` (clothing, accessories) from the user's idea.
* **environment**: Create a fitting `setting` and `background`. Define the `atmosphere`.
* **style**: Choose a `medium` (digital art, photo), `art_style` (fantasy, cyberpunk), and `aesthetic`. Add `art_platforms` like ["trending on artstation"].
* **technical**: Always add `quality_tags` ["masterpiece", "best quality", "highly detailed", "8k", "ultra detailed"] and `resolution_tags` ["sharp focus"]. Define a `camera` shot (medium, close-up) and `lighting` (soft, cinematic).
* **artistic**: Choose a `color_palette` and `mood`. Add relevant `artists` (like ["artgerm", "greg rutkowski"]).
* **modifiers**: Add `additional_details` (particles, sparks) and `sweeteners` ["cinematic composition"].
* **emphasis**: (Optional) Add emphasis (e.g., 1.2) to key terms.

### 4. `negative_prompt` Object
* **Always populate this** with standard, high-quality negative prompts to fix common issues.
* **quality_issues**: ["worst quality", "low quality", "blurry", "jpeg artifacts"]
* **anatomical_issues**: ["bad anatomy", "bad hands", "deformed", "extra limbs", "malformed features"]
* **visual_artifacts**: ["watermark", "text", "signature"]
* **style_exclusions**: (e.g., ["3d render", "realistic"] if the prompt is for a "painting")
* **content_exclusions**: (e.g., ["modern clothing"] for a "fantasy" prompt)
* **custom**: (Optional)

### 5. `parameters` Object
* Use these standard defaults unless the user implies otherwise.
* **steps**: 30
* **sampler**: "DPM++ 2M Karras"
* **cfg_scale**: 7
* **seed**: -1
* **width**: 1024
* **height**: 1024

1.  **Analyze <user_input>.**
2.  **Safety Check**: Is it safe or unsafe?
3.  **If SAFE**: Your response **must** be *only* the fully populated, starting from the 
"safe": "Boolean", object based on the schema above.

here is your structured answer schema, You must follow the structure below and ensure that the structure is complete.
You must ensure the completeness of the entire response, especially from "parameters" to the final "}", and nothing should be missing.:
"safe": "Boolean",
"image_url": "string",
"prompt": {
    "subject": {
      "main": "string",
      "details": ["array", "of", "details"],
      "actions": ["poses", "activities"],
      "physical_attributes": {
        "appearance": "string",
        "clothing": "string",
        "accessories": "string"
      }
    },
    "environment": {
      "setting": "string",
      "background": "string",
      "atmosphere": "string"
    },
    "style": {
      "medium": "string",
      "art_style": ["array", "of", "styles"],
      "aesthetic": ["genre", "theme"],
      "art_platforms": ["artstation", "deviantart"]
    },
    "technical": {
      "quality_tags": ["masterpiece", "best quality"],
      "resolution_tags": ["8k", "highly detailed"],
      "camera": {
        "shot_type": "string",
        "angle": "string",
        "lens": "string",
        "depth_of_field": "string"
      },
      "lighting": {
        "type": "string",
        "quality": "string",
        "direction": "string",
        "time_of_day": "string"
      }
    },
    "artistic": {
      "color_palette": ["colors"],
      "mood": "string",
      "artists": ["artist names"]
    },
    "modifiers": {
      "additional_details": ["array"],
      "sweeteners": ["atmospheric elements"]
    },
    "emphasis": {
      "keyword": "weight as float"
    }
  },
  "negative_prompt": {
    "quality_issues": ["worst quality", "low quality"],
    "anatomical_issues": ["bad anatomy", "bad hands"],
    "visual_artifacts": ["watermark", "text"],
    "style_exclusions": ["styles to avoid"],
    "content_exclusions": ["unwanted elements"],
    "custom": ["additional negatives"]
  },
  "parameters": {
    "steps": "integer",
    "sampler": "string",
    "cfg_scale": "float",
    "seed": "integer or -1",
    "width": "integer",
    "height": "integer"
  }
}


Here is an example of generating a prompt for the user input: "A beautiful elven sorceress casting a spell in a mystical forest."
{
  "safe": true,
  "image_url": "",
  "prompt": {
    "subject": {
      "main": "beautiful elven sorceress",
      "details": [
        "pointed ears",
        "ethereal beauty",
        "long flowing silver hair",
        "glowing blue eyes"
      ],
      "actions": ["casting spell", "hands glowing with magic"],
      "physical_attributes": {
        "appearance": "pale skin, delicate features, mystical aura",
        "clothing": "elegant flowing robes with intricate embroidery, gemstone accents",
        "accessories": "ornate circlet, magical staff"
      }
    },
    "environment": {
      "setting": "ancient mystical forest",
      "background": "glowing mushrooms, ethereal mist, ancient trees",
      "atmosphere": "magical, mysterious, enchanted"
    },
    "style": {
      "medium": "digital art",
      "art_style": ["fantasy art", "concept art"],
      "aesthetic": ["high fantasy", "ethereal"],
      "art_platforms": ["trending on artstation"]
    },
    "technical": {
      "quality_tags": ["masterpiece", "best quality", "highly detailed", "8k"],
      "resolution_tags": ["ultra detailed", "sharp focus"],
      "camera": {
        "shot_type": "medium close-up",
        "angle": "slightly low angle",
        "lens": "portrait lens",
        "depth_of_field": "bokeh background"
      },
      "lighting": {
        "type": "magical lighting",
        "quality": "soft dramatic lighting",
        "direction": "rim lighting",
        "time_of_day": "twilight"
      }
    },
    "artistic": {
      "color_palette": ["purple", "blue", "silver", "iridescent"],
      "mood": "mystical and enchanting",
      "artists": ["artgerm", "ross tran", "wlop"]
    },
    "modifiers": {
      "additional_details": ["magical particles", "glowing runes", "fantasy atmosphere"],
      "sweeteners": ["cinematic composition", "award winning"]
    },
    "emphasis": {
      "glowing magic": 1.3,
      "detailed face": 1.2,
      "ethereal": 1.1
    }
  },
  "negative_prompt": {
    "quality_issues": ["worst quality", "low quality", "blurry", "low res"],
    "anatomical_issues": ["bad anatomy", "bad hands", "deformed", "extra limbs"],
    "visual_artifacts": ["watermark", "text", "signature", "jpeg artifacts"],
    "style_exclusions": ["realistic photo", "3d render"],
    "content_exclusions": ["modern clothing", "technology"],
    "custom": ["masculine features", "harsh lighting"]
  },
  "parameters": {
    "steps": 30,
    "sampler": "DPM++ 2M Karras",
    "cfg_scale": 7.5,
    "seed": -1,
    "width": 896,
    "height": 1152
  }
}


---

Here is the user's request. Begin your analysis.

'''




from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from typing import TypeVar

search_tool = ValyuSearchTool(
    valyu_api_key=os.getenv("VALYU_API_KEY"),
    # search_type="all",  # Search both proprietary and web sources
    max_num_results=5,   # Limit results
    relevance_threshold=0.5,  # Minimum relevance score
    # max_price=20.0  # Maximum cost in dollars
)
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

SchemaError = TypeVar('SchemaError')

class ToolStrategy(Generic[SchemaError]):
    schema: type[Schema]
    tool_message_content: str | None
    handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]

class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None
        message_content = first_message.content
        text_content = ""  # 默认的空文本

        if isinstance(message_content, str):
            text_content = message_content
        elif isinstance(message_content, list):
            # 2. 如果是列表，遍历它来查找第一个 "text" 部分
          for part in message_content:
            if isinstance(part, dict) and part.get("type") == "text":
              text_content = part.get("text", "")
            break  # 找到后就跳出循环

        # 3. 现在再安全地调用 .lower()
        content = text_content.lower()

        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }

        return None

product_review_schema ={
  "safe": "Boolean",
  "prompt": {
    "subject": {
      "main": "string",
      "details": ["array", "of", "details"],
      "actions": ["poses", "activities"],
      "physical_attributes": {
        "appearance": "string",
        "clothing": "string",
        "accessories": "string"
      }
    },
    "environment": {
      "setting": "string",
      "background": "string",
      "atmosphere": "string"
    },
    "style": {
      "medium": "string",
      "art_style": ["array", "of", "styles"],
      "aesthetic": ["genre", "theme"],
      "art_platforms": ["artstation", "deviantart"]
    },
    "technical": {
      "quality_tags": ["masterpiece", "best quality"],
      "resolution_tags": ["8k", "highly detailed"],
      "camera": {
        "shot_type": "string",
        "angle": "string",
        "lens": "string",
        "depth_of_field": "string"
      },
      "lighting": {
        "type": "string",
        "quality": "string",
        "direction": "string",
        "time_of_day": "string"
      }
    },
    "artistic": {
      "color_palette": ["colors"],
      "mood": "string",
      "artists": ["artist names"]
    },
    "modifiers": {
      "additional_details": ["array"],
      "sweeteners": ["atmospheric elements"]
    },
    "emphasis": {
      "keyword": "weight as float"
    }
  },
  "negative_prompt": {
    "quality_issues": ["worst quality", "low quality"],
    "anatomical_issues": ["bad anatomy", "bad hands"],
    "visual_artifacts": ["watermark", "text"],
    "style_exclusions": ["styles to avoid"],
    "content_exclusions": ["unwanted elements"],
    "custom": ["additional negatives"]
  },
  "parameters": {
    "steps": "integer",
    "sampler": "string",
    "cfg_scale": "float",
    "seed": "integer or -1",
    "width": "integer",
    "height": "integer",
    "batch_size": "integer",
    "denoising_strength": "float"
  }
}





agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_tool],
    context_schema=Context,
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=profanity_word
        )],
    checkpointer=checkpointer
)





# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}
import base64


image_path = r'F:\UCL\hackthon\holistic.png'

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"错误：文件未找到于 {image_path}")
    # 在这里停止执行或处理错误
    exit()

# 2. 自动获取 MIME 类型 (对于 .png 应该是 'image/png')
mime_type, _ = mimetypes.guess_type(image_path)
if mime_type is None:
    mime_type = "image/png" # 如果猜不到，手动指定

# 3. 读取文件, 编码为 Base64, 并解码为 UTF-8 字符串
with open(image_path, "rb") as image_file:
    base64_data = base64.b64encode(image_file.read()).decode("utf-8")

# 4. 构建完整的 Data URI
#    这就是将要放入 "url" 字段的内容
data_uri = f"data:{mime_type};base64,{base64_data}"


response = agent.invoke(
  {"messages": {"role": "user",
  "content": [
    {"type": "text", "text": "Generate an image that puts this thing in a shirt."},
    {"type": "image_url",
        "image_url": {
            "url": data_uri  
        }}
  ]}},
  config=config,
  context=Context(user_id="1")
)

import json

# 3. 移除 Markdown 标记
# .strip() 用于去除开头和结尾的空白（包括换行符）
# .removeprefix("```json") 移除开头的标记
# .removesuffix("```") 移除结尾的标记
json_string_cleaned = response['messages'][1].content.strip().removeprefix("```json").removesuffix("```")

print(json_string_cleaned)
if profanity.contains_profanity(json_string_cleaned):

  print("contains_profanity")
else:    
  class PromptSchemaValidator:
    """A validator for Stable Diffusion prompt JSON schemas."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
            strict_mode: If True, unknown fields will be treated as errors.
        """
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
        
    def validate(self, json_string: str) -> tuple[bool, List[str], List[str]]:
        """
        Validate a JSON string against the prompt schema.
        
        Returns:
            tuple: (is_valid, errors_list, warnings_list)
        """
        self.errors = []
        self.warnings = []
        
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {str(e)}")
            return False, self.errors, self.warnings
        
        # SAFETY CHECK: First check if 'safe' field exists and is true
        if 'safe' not in data:
            self.errors.append("SAFETY ERROR: Missing 'safe' field")
            return False, self.errors, self.warnings
        
        if not isinstance(data['safe'], bool):
            self.errors.append(f"SAFETY ERROR: 'safe' field must be boolean, got {type(data['safe']).__name__}")
            return False, self.errors, self.warnings
        
        if data['safe'] is False:
            self.errors.append("SAFETY ERROR: Prompt is marked as unsafe (safe=false). Validation stopped.")
            return False, self.errors, self.warnings
        
        # Check for image_url field (required after safe field)
        if 'image_url' not in data:
            self.errors.append("Missing required field: 'image_url'")
        else:
            self._validate_image_url(data['image_url'])
        
        # Only proceed with full validation if safe=true
        self._validate_structure(data)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_structure(self, data: Dict) -> None:
        """Validate the overall structure of the JSON data."""
        
        # Required top-level fields (safe already checked in validate())
        required_fields = {
            'prompt': (dict, self._validate_prompt),
            'negative_prompt': (dict, self._validate_negative_prompt),
            'parameters': (dict, self._validate_parameters)
        }
        
        # Optional top-level fields
        optional_fields = {
            'advanced': (dict, self._validate_advanced)
        }
        
        # Check required fields
        for field, (expected_type, validator) in required_fields.items():
            if field not in data:
                self.errors.append(f"Missing required field: '{field}'")
            elif not isinstance(data[field], expected_type):
                self.errors.append(f"Field '{field}': Expected {expected_type.__name__}, got {type(data[field]).__name__}")
            elif validator:
                validator(data[field], field)
        
        # Check optional fields
        for field, (expected_type, validator) in optional_fields.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    self.errors.append(f"Field '{field}': Expected {expected_type.__name__}, got {type(data[field]).__name__}")
                elif validator:
                    validator(data[field], field)
        
        # Check for unknown fields
        all_known_fields = set(required_fields.keys()) | set(optional_fields.keys()) | {'safe', 'image_url'}  # 'safe' and 'image_url' are checked earlier
        unknown_fields = set(data.keys()) - all_known_fields
        if unknown_fields:
            if self.strict_mode:
                for field in unknown_fields:
                    self.errors.append(f"Unknown field: '{field}'")
            else:
                for field in unknown_fields:
                    self.warnings.append(f"Unknown field: '{field}' (ignored)")
    
    def _validate_prompt(self, prompt: Dict, path: str) -> None:
        """Validate the prompt section."""
        required_sections = {
            'subject': self._validate_subject,
            'environment': self._validate_environment,
            'style': self._validate_style,
            'technical': self._validate_technical,
            'artistic': self._validate_artistic,
            'modifiers': self._validate_modifiers,
            'emphasis': self._validate_emphasis
        }
        
        for section, validator in required_sections.items():
            if section not in prompt:
                self.errors.append(f"{path}.{section}: Missing required section")
            else:
                validator(prompt[section], f"{path}.{section}")
    
    def _validate_subject(self, subject: Dict, path: str) -> None:
        """Validate the subject section."""
        self._check_string_field(subject, 'main', path)
        self._check_string_list(subject, 'details', path)
        self._check_string_list(subject, 'actions', path)
        
        if 'physical_attributes' not in subject:
            self.errors.append(f"{path}.physical_attributes: Missing required field")
        elif isinstance(subject['physical_attributes'], dict):
            attrs = subject['physical_attributes']
            self._check_string_field(attrs, 'appearance', f"{path}.physical_attributes")
            self._check_string_field(attrs, 'clothing', f"{path}.physical_attributes")
            self._check_string_field(attrs, 'accessories', f"{path}.physical_attributes")
    
    def _validate_environment(self, env: Dict, path: str) -> None:
        """Validate the environment section."""
        self._check_string_field(env, 'setting', path)
        self._check_string_field(env, 'background', path)
        self._check_string_field(env, 'atmosphere', path)
    
    def _validate_style(self, style: Dict, path: str) -> None:
        """Validate the style section."""
        self._check_string_field(style, 'medium', path)
        self._check_string_list(style, 'art_style', path)
        self._check_string_list(style, 'aesthetic', path)
        self._check_string_list(style, 'art_platforms', path)
    
    def _validate_technical(self, technical: Dict, path: str) -> None:
        """Validate the technical section."""
        self._check_string_list(technical, 'quality_tags', path)
        self._check_string_list(technical, 'resolution_tags', path)
        
        # Validate camera subsection
        if 'camera' not in technical:
            self.errors.append(f"{path}.camera: Missing required field")
        elif isinstance(technical['camera'], dict):
            camera = technical['camera']
            self._check_string_field(camera, 'shot_type', f"{path}.camera")
            self._check_string_field(camera, 'angle', f"{path}.camera")
            self._check_string_field(camera, 'lens', f"{path}.camera")
            self._check_string_field(camera, 'depth_of_field', f"{path}.camera")
        
        # Validate lighting subsection
        if 'lighting' not in technical:
            self.errors.append(f"{path}.lighting: Missing required field")
        elif isinstance(technical['lighting'], dict):
            lighting = technical['lighting']
            self._check_string_field(lighting, 'type', f"{path}.lighting")
            self._check_string_field(lighting, 'quality', f"{path}.lighting")
            self._check_string_field(lighting, 'direction', f"{path}.lighting")
            self._check_string_field(lighting, 'time_of_day', f"{path}.lighting")
    
    def _validate_artistic(self, artistic: Dict, path: str) -> None:
        """Validate the artistic section."""
        self._check_string_list(artistic, 'color_palette', path)
        self._check_string_field(artistic, 'mood', path)
        self._check_string_list(artistic, 'artists', path)
    
    def _validate_modifiers(self, modifiers: Dict, path: str) -> None:
        """Validate the modifiers section."""
        self._check_string_list(modifiers, 'additional_details', path)
        self._check_string_list(modifiers, 'sweeteners', path)
    
    def _validate_emphasis(self, emphasis: Dict, path: str) -> None:
        """Validate the emphasis section (dynamic keys with float values)."""
        if not isinstance(emphasis, dict):
            self.errors.append(f"{path}: Expected dict, got {type(emphasis).__name__}")
            return
        
        for key, value in emphasis.items():
            if not isinstance(value, (int, float)):
                self.errors.append(f"{path}.{key}: Expected numeric weight, got {type(value).__name__}")
    
    def _validate_negative_prompt(self, negative: Dict, path: str) -> None:
        """Validate the negative prompt section."""
        required_fields = [
            'quality_issues',
            'anatomical_issues',
            'visual_artifacts',
            'style_exclusions',
            'content_exclusions',
            'custom'
        ]
        
        for field in required_fields:
            self._check_string_list(negative, field, path)
    
    def _validate_parameters(self, params: Dict, path: str) -> None:
        """Validate the parameters section."""
        validations = {
            'steps': (int, lambda x: x > 0),
            'sampler': (str, None),
            'cfg_scale': ((int, float), lambda x: x >= 0),
            'seed': (int, lambda x: x >= -1),
            'width': (int, lambda x: x > 0),
            'height': (int, lambda x: x > 0),
        }
        
        for field, (expected_type, validator_func) in validations.items():
            if field == 'denoising_strength':  # Optional field
                if field in params:
                    self._validate_param_field(params, field, expected_type, validator_func, path)
            else:  # Required fields
                if field not in params:
                    self.errors.append(f"{path}.{field}: Missing required field")
                else:
                    self._validate_param_field(params, field, expected_type, validator_func, path)
    
    def _validate_image_url(self, image_url: Any) -> None:
        """
        Validate the image_url field containing a data URI.
        
        A valid data URI should follow the format:
        data:[<mediatype>][;base64],<data>
        
        Examples:
        - data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...
        - data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...
        - data:text/plain;charset=UTF-8,Hello%20World
        """
        import re
        
        if not isinstance(image_url, str):
            self.errors.append(f"image_url: Expected string (data URI), got {type(image_url).__name__}")
            return
        
        # Check if it's empty
        if not image_url:
            self.errors.append("image_url: Cannot be empty")
            return
        
        # Basic data URI pattern validation
        # Format: data:[<mediatype>][;base64],<data>
        data_uri_pattern = r'^data:([a-zA-Z0-9][a-zA-Z0-9\/\+\-]*)(;[a-zA-Z0-9\-]+=[a-zA-Z0-9\-]+)*(;base64)?,'
        
        if not re.match(data_uri_pattern, image_url):
            self.errors.append("image_url: Invalid data URI format. Expected format: data:[<mediatype>][;base64],<data>")
            return
        
        # Check for common image MIME types if it's supposed to be an image
        image_mime_types = [
            'image/jpeg',
            'image/jpg', 
            'image/png',
            'image/gif',
            'image/webp',
            'image/bmp',
            'image/svg+xml'
        ]
        
        # Extract MIME type from data URI
        mime_match = re.match(r'^data:([^;,]+)', image_url)
        if mime_match:
            mime_type = mime_match.group(1)
            if not any(mime_type.startswith(img_type) for img_type in image_mime_types):
                self.warnings.append(f"image_url: Unusual MIME type '{mime_type}'. Expected image type (e.g., image/jpeg, image/png)")
        
        # Check if base64 encoded (most common for images)
        if ';base64,' in image_url:
            # Validate base64 content
            base64_data = image_url.split(';base64,', 1)[1]
            if not base64_data:
                self.errors.append("image_url: Base64 data portion is empty")
            else:
                # Check if base64 string is valid (basic check)
                import base64
                try:
                    # Try to decode a small portion to verify it's valid base64
                    test_portion = base64_data[:100]  # Test first 100 chars
                    base64.b64decode(test_portion + '=' * (4 - len(test_portion) % 4))
                except Exception:
                    self.errors.append("image_url: Invalid base64 encoding in data URI")
        
        # Check reasonable size (warn if too large)
        if len(image_url) > 5_000_000:  # 5MB as base64 is roughly 5 million characters
            self.warnings.append("image_url: Very large data URI (>5MB). Consider using external image hosting.")
        elif len(image_url) < 50:
            self.errors.append("image_url: Data URI too short to contain valid image data")
    
    def _validate_param_field(self, params: Dict, field: str, expected_type: type, 
                              validator_func: Optional[callable], path: str) -> None:
        """Validate a parameter field."""
        value = params[field]
        if not isinstance(value, expected_type):
            self.errors.append(f"{path}.{field}: Expected {expected_type}, got {type(value).__name__}")
        elif validator_func and not validator_func(value):
            self.errors.append(f"{path}.{field}: Value {value} failed validation")
    
    def _check_string_field(self, data: Dict, field: str, path: str) -> None:
        """Check if a field exists and is a string."""
        if field not in data:
            self.errors.append(f"{path}.{field}: Missing required field")
        elif not isinstance(data[field], str):
            self.errors.append(f"{path}.{field}: Expected string, got {type(data[field]).__name__}")
    
    def _check_string_list(self, data: Dict, field: str, path: str) -> None:
        """Check if a field exists and is a list of strings."""
        if field not in data:
            self.errors.append(f"{path}.{field}: Missing required field")
        elif not isinstance(data[field], list):
            self.errors.append(f"{path}.{field}: Expected list, got {type(data[field]).__name__}")
        else:
            for i, item in enumerate(data[field]):
                if not isinstance(item, str):
                    self.errors.append(f"{path}.{field}[{i}]: Expected string, got {type(item).__name__}")
      
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
          strict_mode: If True, unknown fields will be treated as errors.
        """
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
          
    def validate(self, json_string: str) -> tuple[bool, List[str], List[str]]:
          """
          Validate a JSON string against the prompt schema.
          
          Returns:
              tuple: (is_valid, errors_list, warnings_list)
          """
          self.errors = []
          self.warnings = []
          
          try:
              data = json.loads(json_string)
          except json.JSONDecodeError as e:
              self.errors.append(f"Invalid JSON: {str(e)}")
              return False, self.errors, self.warnings
          
          # SAFETY CHECK: First check if 'safe' field exists and is true
          if 'safe' not in data:
              self.errors.append("SAFETY ERROR: Missing 'safe' field")
              return False, self.errors, self.warnings
          
          if not isinstance(data['safe'], bool):
              self.errors.append(f"SAFETY ERROR: 'safe' field must be boolean, got {type(data['safe']).__name__}")
              return False, self.errors, self.warnings
          
          if data['safe'] is False:
              self.errors.append("SAFETY ERROR: Prompt is marked as unsafe (safe=false). Validation stopped.")
              return False, self.errors, self.warnings
          
          # Only proceed with full validation if safe=true
          self._validate_structure(data)
          
          return len(self.errors) == 0, self.errors, self.warnings
      
    def _validate_structure(self, data: Dict) -> None:
          """Validate the overall structure of the JSON data."""
          
          # Required top-level fields (safe already checked in validate())
          required_fields = {
              'prompt': (dict, self._validate_prompt),
              'negative_prompt': (dict, self._validate_negative_prompt),
              'parameters': (dict, self._validate_parameters)
          }
          
          # Optional top-level fields
          optional_fields = {
              'advanced': (dict, self._validate_advanced)
          }
          
          # Check required fields
          for field, (expected_type, validator) in required_fields.items():
              if field not in data:
                  self.errors.append(f"Missing required field: '{field}'")
              elif not isinstance(data[field], expected_type):
                  self.errors.append(f"Field '{field}': Expected {expected_type.__name__}, got {type(data[field]).__name__}")
              elif validator:
                  validator(data[field], field)
          
          # Check optional fields
          for field, (expected_type, validator) in optional_fields.items():
              if field in data:
                  if not isinstance(data[field], expected_type):
                      self.errors.append(f"Field '{field}': Expected {expected_type.__name__}, got {type(data[field]).__name__}")
                  elif validator:
                      validator(data[field], field)
          
          # Check for unknown fields
          all_known_fields = set(required_fields.keys()) | set(optional_fields.keys()) | {'safe'}  # 'safe' is checked earlier
          unknown_fields = set(data.keys()) - all_known_fields
          if unknown_fields:
              if self.strict_mode:
                  for field in unknown_fields:
                      self.errors.append(f"Unknown field: '{field}'")
              else:
                  for field in unknown_fields:
                      self.warnings.append(f"Unknown field: '{field}' (ignored)")
      
    def _validate_prompt(self, prompt: Dict, path: str) -> None:
          """Validate the prompt section."""
          required_sections = {
              'subject': self._validate_subject,
              'environment': self._validate_environment,
              'style': self._validate_style,
              'technical': self._validate_technical,
              'artistic': self._validate_artistic,
              'modifiers': self._validate_modifiers,
              'emphasis': self._validate_emphasis
          }
          
          for section, validator in required_sections.items():
              if section not in prompt:
                  self.errors.append(f"{path}.{section}: Missing required section")
              else:
                  validator(prompt[section], f"{path}.{section}")
      
    def _validate_subject(self, subject: Dict, path: str) -> None:
          """Validate the subject section."""
          self._check_string_field(subject, 'main', path)
          self._check_string_list(subject, 'details', path)
          self._check_string_list(subject, 'actions', path)
          
          if 'physical_attributes' not in subject:
              self.errors.append(f"{path}.physical_attributes: Missing required field")
          elif isinstance(subject['physical_attributes'], dict):
              attrs = subject['physical_attributes']
              self._check_string_field(attrs, 'appearance', f"{path}.physical_attributes")
              self._check_string_field(attrs, 'clothing', f"{path}.physical_attributes")
              self._check_string_field(attrs, 'accessories', f"{path}.physical_attributes")
      
    def _validate_environment(self, env: Dict, path: str) -> None:
          """Validate the environment section."""
          self._check_string_field(env, 'setting', path)
          self._check_string_field(env, 'background', path)
          self._check_string_field(env, 'atmosphere', path)
      
    def _validate_style(self, style: Dict, path: str) -> None:
          """Validate the style section."""
          self._check_string_field(style, 'medium', path)
          self._check_string_list(style, 'art_style', path)
          self._check_string_list(style, 'aesthetic', path)
          self._check_string_list(style, 'art_platforms', path)
      
    def _validate_technical(self, technical: Dict, path: str) -> None:
          """Validate the technical section."""
          self._check_string_list(technical, 'quality_tags', path)
          self._check_string_list(technical, 'resolution_tags', path)
          
          # Validate camera subsection
          if 'camera' not in technical:
              self.errors.append(f"{path}.camera: Missing required field")
          elif isinstance(technical['camera'], dict):
              camera = technical['camera']
              self._check_string_field(camera, 'shot_type', f"{path}.camera")
              self._check_string_field(camera, 'angle', f"{path}.camera")
              self._check_string_field(camera, 'lens', f"{path}.camera")
              self._check_string_field(camera, 'depth_of_field', f"{path}.camera")
          
          # Validate lighting subsection
          if 'lighting' not in technical:
              self.errors.append(f"{path}.lighting: Missing required field")
          elif isinstance(technical['lighting'], dict):
              lighting = technical['lighting']
              self._check_string_field(lighting, 'type', f"{path}.lighting")
              self._check_string_field(lighting, 'quality', f"{path}.lighting")
              self._check_string_field(lighting, 'direction', f"{path}.lighting")
              self._check_string_field(lighting, 'time_of_day', f"{path}.lighting")
      
    def _validate_artistic(self, artistic: Dict, path: str) -> None:
          """Validate the artistic section."""
          self._check_string_list(artistic, 'color_palette', path)
          self._check_string_field(artistic, 'mood', path)
          self._check_string_list(artistic, 'artists', path)
      
    def _validate_modifiers(self, modifiers: Dict, path: str) -> None:
          """Validate the modifiers section."""
          self._check_string_list(modifiers, 'additional_details', path)
          self._check_string_list(modifiers, 'sweeteners', path)
      
    def _validate_emphasis(self, emphasis: Dict, path: str) -> None:
          """Validate the emphasis section (dynamic keys with float values)."""
          if not isinstance(emphasis, dict):
              self.errors.append(f"{path}: Expected dict, got {type(emphasis).__name__}")
              return
          
          for key, value in emphasis.items():
              if not isinstance(value, (int, float)):
                  self.errors.append(f"{path}.{key}: Expected numeric weight, got {type(value).__name__}")
      
    def _validate_negative_prompt(self, negative: Dict, path: str) -> None:
          """Validate the negative prompt section."""
          required_fields = [
              'quality_issues',
              'anatomical_issues',
              'visual_artifacts',
              'style_exclusions',
              'content_exclusions',
              'custom'
          ]
          
          for field in required_fields:
              self._check_string_list(negative, field, path)
      
    def _validate_parameters(self, params: Dict, path: str) -> None:
          """Validate the parameters section."""
          validations = {
              'steps': (int, lambda x: x > 0),
              'sampler': (str, None),
              'cfg_scale': ((int, float), lambda x: x >= 0),
              'seed': (int, lambda x: x >= -1),
              'width': (int, lambda x: x > 0),
              'height': (int, lambda x: x > 0)
          }
          
          for field, (expected_type, validator_func) in validations.items():
              if field == 'denoising_strength':  # Optional field
                  if field in params:
                      self._validate_param_field(params, field, expected_type, validator_func, path)
              else:  # Required fields
                  if field not in params:
                      self.errors.append(f"{path}.{field}: Missing required field")
                  else:
                      self._validate_param_field(params, field, expected_type, validator_func, path)
      
    def _validate_advanced(self, advanced: Dict, path: str) -> None:
          """Validate the optional advanced section."""
          # This is optional and flexible, so we just check basic structure
          if 'loras' in advanced and not isinstance(advanced['loras'], list):
              self.errors.append(f"{path}.loras: Expected list, got {type(advanced['loras']).__name__}")
          
          if 'embeddings' in advanced and not isinstance(advanced['embeddings'], dict):
              self.errors.append(f"{path}.embeddings: Expected dict, got {type(advanced['embeddings']).__name__}")
          
          if 'controlnet' in advanced and not isinstance(advanced['controlnet'], list):
              self.errors.append(f"{path}.controlnet: Expected list, got {type(advanced['controlnet']).__name__}")
          
          if 'regional_prompting' in advanced and not isinstance(advanced['regional_prompting'], dict):
              self.errors.append(f"{path}.regional_prompting: Expected dict, got {type(advanced['regional_prompting']).__name__}")
      
    def _validate_param_field(self, params: Dict, field: str, expected_type: type, 
                                validator_func: Optional[callable], path: str) -> None:
          """Validate a parameter field."""
          value = params[field]
          if not isinstance(value, expected_type):
              self.errors.append(f"{path}.{field}: Expected {expected_type}, got {type(value).__name__}")
          elif validator_func and not validator_func(value):
              self.errors.append(f"{path}.{field}: Value {value} failed validation")
      
    def _check_string_field(self, data: Dict, field: str, path: str) -> None:
          """Check if a field exists and is a string."""
          if field not in data:
              self.errors.append(f"{path}.{field}: Missing required field")
          elif not isinstance(data[field], str):
              self.errors.append(f"{path}.{field}: Expected string, got {type(data[field]).__name__}")
      
    def _check_string_list(self, data: Dict, field: str, path: str) -> None:
          """Check if a field exists and is a list of strings."""
          if field not in data:
              self.errors.append(f"{path}.{field}: Missing required field")
          elif not isinstance(data[field], list):
              self.errors.append(f"{path}.{field}: Expected list, got {type(data[field]).__name__}")
          else:
              for i, item in enumerate(data[field]):
                  if not isinstance(item, str):
                      self.errors.append(f"{path}.{field}[{i}]: Expected string, got {type(item).__name__}")




      



  validator = PromptSchemaValidator(strict_mode=False)
  is_valid, errors, warnings = validator.validate(json_string_cleaned)
  print(f"Validation Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
  if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
              print(f"  ❌ {error}")
      
  if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
      
  if is_valid:
        print("\n✨ The JSON perfectly conforms to the specified schema!")
        data = json.loads(json_string_cleaned)
        file_name = "structured prompt.json"
        with open(file_name, 'w', encoding='utf-8') as f:
          json.dump(data, f, ensure_ascii=False, indent=4)     
        print(f"Success, the structured prompt has been successfully written.: {file_name}")




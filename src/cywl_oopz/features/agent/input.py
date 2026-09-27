"""Compatibility imports for the shared user input contract."""

from cywl_oopz.conversation.input import IMAGE_ONLY_PROMPT as IMAGE_ONLY_PROMPT
from cywl_oopz.conversation.input import ImageInputPart as ImageInputPart
from cywl_oopz.conversation.input import InputPart as InputPart
from cywl_oopz.conversation.input import TextInputPart as TextInputPart
from cywl_oopz.conversation.input import UserInput as AgentUserInput

__all__ = ["AgentUserInput", "IMAGE_ONLY_PROMPT", "ImageInputPart", "InputPart", "TextInputPart"]

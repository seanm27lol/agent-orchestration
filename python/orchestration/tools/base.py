"""
Tool Base Classes

Provides abstractions for creating tools that agents can invoke.
Follows a decorator-based pattern for easy tool creation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Awaitable, Any, Optional, Dict, List, Union
from enum import Enum
from datetime import datetime
import uuid
import functools
import inspect
import asyncio


# Type aliases
ToolHandler = Callable[..., Awaitable[Any]]
T = TypeVar('T')


class ToolCategory(Enum):
    """Categories for organizing tools."""
    WEB = "web"
    CODE = "code"
    DATA = "data"
    FILE = "file"
    COMMUNICATION = "communication"
    SEARCH = "search"
    UTILITY = "utility"
    CUSTOM = "custom"


class ToolError(Exception):
    """Base error for tool execution."""

    def __init__(self, code: str, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None

    def validate(self, value: Any) -> bool:
        """Validate a value against this parameter."""
        if value is None:
            return not self.required

        type_checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
        }

        checker = type_checks.get(self.type, lambda v: True)
        return checker(value)


@dataclass
class ToolContext:
    """Context provided when a tool is invoked."""
    agent_id: str
    execution_id: Optional[str] = None
    workflow_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    call_id: str
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    duration_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolDefinition:
    """Complete definition of a tool."""
    name: str
    description: str
    category: str
    parameters: List[ToolParameter]
    returns_type: str = "object"
    returns_description: str = "Result"
    examples: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """
    Abstract base class for tools.

    Subclass this to create custom tools:

    ```python
    class CalculatorTool(Tool):
        def __init__(self):
            super().__init__(
                name="calculator",
                description="Evaluate math expressions",
                category=ToolCategory.UTILITY,
                parameters=[
                    ToolParameter("expression", "string", "Math expression", required=True)
                ]
            )

        async def execute(self, expression: str, context: ToolContext) -> float:
            return eval(expression)  # Use safe evaluation in practice!
    ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: Union[str, ToolCategory] = ToolCategory.CUSTOM,
        parameters: Optional[List[ToolParameter]] = None,
        timeout_ms: int = 30000,
    ):
        self.name = name
        self.description = description
        self.category = category.value if isinstance(category, ToolCategory) else category
        self.parameters = parameters or []
        self.timeout_ms = timeout_ms

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            category=self.category,
            parameters=self.parameters,
        )

    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        for param in self.parameters:
            value = kwargs.get(param.name)

            if value is None and param.required:
                raise ToolError(
                    "INVALID_INPUT",
                    f"Required parameter '{param.name}' is missing"
                )

            if value is not None and not param.validate(value):
                raise ToolError(
                    "INVALID_INPUT",
                    f"Parameter '{param.name}' has invalid type, expected {param.type}"
                )

    @abstractmethod
    async def execute(self, context: ToolContext, **kwargs) -> Any:
        """
        Execute the tool.

        Override this method to implement tool logic.

        Args:
            context: Execution context with agent info
            **kwargs: Tool parameters

        Returns:
            Tool result (any JSON-serializable value)
        """
        pass

    async def invoke(
        self,
        context: ToolContext,
        call_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Invoke the tool with full result tracking.

        Args:
            context: Execution context
            call_id: Optional call ID (generated if not provided)
            **kwargs: Tool parameters

        Returns:
            ToolResult with success/error info
        """
        call_id = call_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            # Validate input
            self.validate_input(**kwargs)

            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute(context, **kwargs),
                timeout=self.timeout_ms / 1000
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                result=result,
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error={
                    "code": "TIMEOUT",
                    "message": f"Tool execution timed out after {self.timeout_ms}ms"
                },
                duration_ms=duration,
            )

        except ToolError as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error={
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                },
                duration_ms=duration,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error={
                    "code": "EXECUTION_ERROR",
                    "message": str(e),
                },
                duration_ms=duration,
            )


class FunctionTool(Tool):
    """
    Tool wrapper for async functions.

    Created by the @tool decorator.
    """

    def __init__(
        self,
        func: ToolHandler,
        name: str,
        description: str,
        category: Union[str, ToolCategory],
        parameters: List[ToolParameter],
        timeout_ms: int,
    ):
        super().__init__(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            timeout_ms=timeout_ms,
        )
        self._func = func

    async def execute(self, context: ToolContext, **kwargs) -> Any:
        # Check if function accepts context
        sig = inspect.signature(self._func)
        if 'context' in sig.parameters:
            return await self._func(context=context, **kwargs)
        return await self._func(**kwargs)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Union[str, ToolCategory] = ToolCategory.CUSTOM,
    timeout_ms: int = 30000,
):
    """
    Decorator for creating tools from async functions.

    Parameters are automatically extracted from the function signature.
    Type hints are used for parameter types.

    ```python
    @tool(
        name="calculator",
        description="Evaluate math expressions",
        category=ToolCategory.UTILITY
    )
    async def calculate(expression: str) -> float:
        '''
        Evaluate a mathematical expression.

        Args:
            expression: The math expression to evaluate
        '''
        return eval(expression)
    ```
    """

    def decorator(func: ToolHandler) -> FunctionTool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Extract parameters from function signature
        sig = inspect.signature(func)
        params: List[ToolParameter] = []

        for param_name, param in sig.parameters.items():
            if param_name == 'context':
                continue  # Skip context parameter

            # Determine type from annotation
            param_type = "object"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "number",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                param_type = type_map.get(param.annotation, "object")

            # Determine if required
            required = param.default == inspect.Parameter.empty

            params.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
                default=None if required else param.default,
            ))

        return FunctionTool(
            func=func,
            name=tool_name,
            description=tool_desc,
            category=category,
            parameters=params,
            timeout_ms=timeout_ms,
        )

    return decorator


# ============================================================================
# Example Tools
# ============================================================================

@tool(
    name="echo",
    description="Echo back the input message",
    category=ToolCategory.UTILITY
)
async def echo_tool(message: str) -> str:
    """Echo the input message back."""
    return f"Echo: {message}"


@tool(
    name="add_numbers",
    description="Add two numbers together",
    category=ToolCategory.UTILITY
)
async def add_numbers_tool(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool(
    name="format_text",
    description="Format text with various transformations",
    category=ToolCategory.UTILITY
)
async def format_text_tool(
    text: str,
    operation: str = "uppercase"
) -> str:
    """
    Format text using the specified operation.

    Args:
        text: The text to format
        operation: The operation (uppercase, lowercase, capitalize)
    """
    operations = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "capitalize": str.capitalize,
    }

    func = operations.get(operation)
    if not func:
        raise ToolError("INVALID_INPUT", f"Unknown operation: {operation}")

    return func(text)


# Export example tools
example_tools = [echo_tool, add_numbers_tool, format_text_tool]

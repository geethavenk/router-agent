import os
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


load_dotenv()

## LLM

llm = ChatOpenAI(
    model="openai/gpt-oss-120b-Turbo",
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],
    temperature=0
)

## Structured routing output

class RouteDecision(BaseModel):
    route: Literal[
        "technical",
        "course",
        "billing",
        "general"
    ]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

router_llm = llm.with_structured_output(RouteDecision)

# Graph state

class AgentState(TypedDict):
    query: str
    route: str
    confidence: float
    final_route: str
    response: str

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.70

# Route agent
def router_node(state: AgentState):
    query = state["query"]
    prompt = f"""
You are a routing agent for an online education platform.

Your job is to classify the user's request into exactly
one of the following departments.

--------------------------------------------------
TECHNICAL
--------------------------------------------------

Use "technical" for:

- Video player problems
- Video buffering
- Login problems
- Website errors
- Application errors
- Technical access problems
- Platform not working

--------------------------------------------------
COURSE
--------------------------------------------------

Use "course" for:

- Course recommendations
- Learning paths
- Which course to choose
- Course content questions
- Learning plans
- Learning Generative AI
- Career-oriented course recommendations

--------------------------------------------------
BILLING
--------------------------------------------------

Use "billing" for:

- Payments
- Failed payments
- Refunds
- Charges
- Subscriptions
- Billing problems
- Payment deducted but course not activated
- Duplicate charges

--------------------------------------------------
GENERAL
--------------------------------------------------

Use "general" when the request does not clearly
belong to the other categories.

--------------------------------------------------

USER QUERY:

{query}

Return:

1. The most appropriate route
2. A confidence score between 0 and 1

The confidence should represent how certain you are
that the selected route is correct.
"""

    decision = router_llm.invoke(prompt)
    final = "general" if decision.confidence < CONFIDENCE_THRESHOLD else decision.route

    return {
        "route": decision.route,                # what the LLM predicted
        "confidence": decision.confidence,
        "final_route": final                    # what will acutally run

    }

# Routing logic

def route_request(state:AgentState):
    return state["final_route"]

# Technical support agent

def technical_agent(state: AgentState):
    query = state["query"]

    prompt = f"""
You are a Technical Support Agent for an
online education platform.

Help the user solve their technical problem.

User question:

{query}

Provide:

- A clear explanation
- Practical troubleshooting steps
- A concise response

If the issue cannot be resolved with basic
troubleshooting, recommend contacting technical support.

Do not invent information about the user's account.

"""
    response = llm.invoke(prompt)

    return {
        "response": response.content,
    }


# Course advisor agent

def course_agent(state: AgentState):
    query = state["query"]

    prompt = f"""
You are a Course Advisor for an online
education platform.

Help users choose courses and learning paths.

User question:

{query}

Provide:

- A useful course recommendation
- A suggested learning path when appropriate
- A concise explanation

If more information about the user's background
or goals is required, ask a short follow-up question.

Do not invent specific courses that do not exist.
"""
    response = llm.invoke(prompt)

    return {
        "response": response.content,
        
    }

# Payment or Billing agent

def billing_agent(state: AgentState):
    query = state["query"]

    prompt = f"""
You are a Payment and Billing Support Agent
for an online education platform.

Handle questions related to:

- Payments
- Failed payments
- Refunds
- Subscriptions
- Billing
- Charges
- Duplicate payments
- Course activation after payment

User question:

{query}

Provide a clear and helpful response.

Do not invent transaction information.

If account-specific investigation is required,
ask the user to contact billing support.
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content,
        

    }

# General support agent

def general_agent(state: AgentState):
    query = state["query"]

    prompt= f"""
You are a General Support Agent for an
online education platform.

Help users whose questions do not clearly
belong to Technical Support, Course Advising,
or Billing.

User question:

{query}

Provide a helpful and concise response.

If another department would be better suited
to handle the issue, explain which department
the user should contact.
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content,
        
    }


# Build langgraph

builder = StateGraph(AgentState)

# Add nodes

builder.add_node("router", router_node)

builder.add_node("technical", technical_agent)

builder.add_node("course", course_agent)

builder.add_node("billing", billing_agent)

builder.add_node("general", general_agent)

# Add edges

builder.add_edge(START, "router")

# router -> scpecialist
builder.add_conditional_edges(
    "router",
    route_request,
    {
        "technical": "technical",
        "course": "course",
        "billing": "billing",
        "general": "general"
    }
)

builder.add_edge("technical", END)
builder.add_edge("course", END)
builder.add_edge("billing", END)
builder.add_edge("general", END)

# compile graph
graph = builder.compile()
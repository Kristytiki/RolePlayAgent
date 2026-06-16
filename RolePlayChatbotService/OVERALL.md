## Intro

Welcome to your take-home/on-site! You will be completing a series of programming tasks. Any code that you produce should be committed to a private git repository and shared with us.

These tasks are purposefully simple and open-ended to give engineers the space to demonstrate their talent. Engineers with front-end talent have built amazing UI’s. Backend engineers have created elegant backends! Use this as a chance to have fun, and build something you are proud of.

* High engineering standards are central to our coding culture.
* We evaluate your code based on elegance, simplicity and creativity. 
* **The current failure rate for this assessment is > 80%**

## Task: Interfacing with CHAI’s model

In the appendix below, we have provided you with API access to one of CHAI’s models. Your tasks will revolve interacting with this API to produce responses for chatbots.

## Task 1: Implement a chatbot [~1-2 hours]

Using the provided model API, implement an interface which allows a user to interact with a chatbot.

## Front End Path

### Task 2: Implement an GUI [~2-3 hours]

### Task 3: Implement an AI chatroom / OR Some other features [~1-2 hours]

Building upon your previous task, implement the ability for two chatbots to engage in a conversation with each other.

---

## Full Stack Path

### Task 2: Implement an AI chatroom [~0.75-1.5 hours]

Building upon your previous task, implement the ability for two chatbots to engage in a conversation with each other.

### Task 3: Productionize your MVP [2.5 hours+]

Now our goal is to take your MVP and serve it to real world users. You will have to make judgement calls on the right approach and trade offs.

1. Decide, and write down what the requirements should be, and your reasoning.
2. Write a plan for how your product should be deployed to real world users.
3. Implement your plan.

---

## Appendix: Model API

The following describes a POST endpoint exposing a stateless API to access one of our models via a chat interface.

> POST
http://guanaco-submitter.guanaco-backend.k2.chaiverse.com/endpoints/onsite/chat

Headers:
Authorization: "Bearer CR_14d43f2bf78b4b0590c2a8b87f354746"

JSON BODY:
{
    "memory": str,
    "prompt": str,
    "bot_name": str,
    "user_name": str,
    "chat_history": List[dict[str, str]]
}

Parameter explanations:
- memory: ** deprecated **
- prompt: **deprecated **
- bot_name: “Einstein”
- user_name: This is the name assigned to agent interacting
             with the model.
- chat_history: This is a list of messages representing the
                conversation history. The format of each
                message is
                {
                    "sender": "sender_name",
                    "message": "some_string"
                }.

Example payload:
{
    "memory": "",
    "prompt": "”,
    # when in QA mode - we suggest bot_name “Einstein” 
    "bot_name": "Einstein",
    "user_name": "User",
    "chat_history":
        [
          # safety prompt - please keep
          {"sender": "Bot", "message": "Please avoid using profanity, or being rude. Be
                          courteous and use language which is appropriate for any audience."},
          {"sender": "User", "message": "Alright"}, 
          # 
          {"sender": "Bot", "message": "Hi there"},
          {"sender": "User", "message": "Hey Bot!"}
        ]
}
> 

## FAQ

Q: Can I use AI?

A: Produce the highest quality work possible. If you choose to use AI please let us know which AI and IDE you used, and a brief description how you used it. E.g. Claude CLI / GPT-5 Codex / etc.

Q. Which language shall I use?

A: For the front-end/GUI use whatever you feel most productive. Backend must be Python.

Q: Can you clarify what you mean by a feature? Is it okay if I do X?
A: The purpose of this task is to demonstrate your strengths. Ultimately we will be rating you based on your outcomes. Did you build something good? Everyone has weaknesses, you will not be scored down on your weaknesses.

Q: I’m not familiar with X, for that reason I’m quite slow.
A: If you feel especially out of your experience, just write a comment. Typically we’d try to factor in personal experience with expectations.

Q: Who shall I reach out to if I have a question?
A: Feel free to ask will@chai-research.com, or anyone else at CHAI. It is better to ask quickly than waste 60 minutes down a rabbit hole. 

Q: About the chat history field. Is there any limit on the input size of chat history? Does it need to be truncated as conversation progresses?
A: It is truncated on our-side, typically at around 4k input-tokens - which is around 40-60 conversational turns.
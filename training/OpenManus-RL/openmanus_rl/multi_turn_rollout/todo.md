You are an expert agent operating in the ALFRED embodied Environment. Your task is
to: {task_description}. Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you
took: {action_history}. You are now at step {current_step} and your current observation
is: {current_observation}. Your admissible actions of the current situation are: [{admissible_actions}].
Now it’s your turn to take an action. You should first reason step-by-step about the current
situation. This reasoning process MUST be enclosed within <think> </think> tags.< + xxxx; how to do the relection, how to do the memory analysis+ how to do general planning  >

Once
you’ve finished your reasoning, you should choose an admissible action for current step and
present it within <action> </action> 
<within action we need action choices and action parameters> 
tags.


<think>
plan
We current in the 
<relfction>
last obs analysis, if we solve the question already?
</reflection>

<memory analysis>
rag style [round 1 thinking], 

[round2 obs]
xxxx
milestones detect?
</memory analysis>

Future plan, next action plan verification
</think>

<action>
+ give answer?
action choices: xxx
action parameters: {'xxx': yyy, xxx}
</action>
<action_result>

</action_result>
x N

Multurn-turn:
prompt->think->act->memory->execute->think->act

1. need to compare modular rollout > reasoning + act + obs; 1-2 point positive

1. add obs?
2. do we make the chat template?
3. masking the obs 
4. let's use the reason-obs one time parse
5. put all the reflection planning, memory use(RAG style generation) into the
6. how to define ending? 

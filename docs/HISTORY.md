# Project history

WorldZero began as a small research environment for asking whether an agent could rearrange persistent objects, produce a causal environmental effect, and leave a useful arrangement for a fresh successor. Early releases established continuous simulated time, material accounting, observation-only policies, matched interventions, replay, and a local observatory.

The R5 and R6 work were historical research and pilot iterations. They helped expose integration, causal-attribution, and state-tracking bottlenecks, but they are not released benchmark families and are not evidence that a language model discovered a tool. Scripted controllers, mock endpoints, incomplete episodes, and mechanical screens remain engineering or pilot evidence only.

Version 0.3 extracted hidden mechanisms behind a typed `LawFamily` interface while retaining one fixed substrate and kernel. Four built-ins now exercise different causal structures, and community packages can add experimental families through exact entry points. Evaluation remains centrally scored and official status remains tied to a reviewed registry identity.

Compact historical reference records are documented in [evidence/reference](../evidence/reference/README.md). The public test suite retains minimal legacy state and trace fixtures solely to verify backward replay compatibility.

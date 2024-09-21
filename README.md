# llm-server

Trying to locally serve an LLM with an OpenAI-like API.

## Usage

To spin up the server, run
```bash
$ rye sync
$ rye run llm-server
```
Then, you could check the API by curl like
```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Tell me the answer to the ultimate question of life, the universe, and everything."}' http://localhost:5000/generate
```


## Design choices

- Since my GPU (GTX 1070) is quite old I decided to use Gemma-2-2B-instruction. This is because its compute capability (CC) is 6.1, which is way less than CC == 7.0 required by triton compiler and vllm.

- When using flask, it executes the main script twice. So I had trouble to blowing up GPU memory by loading model twice in the global name space.

# truss example

```
# pip install baseten truss
# baseten.login()

import baseten
import truss

tr = truss.load("./")
baseten.deploy(tr, model_name="template", publish=True)
```

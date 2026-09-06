# Public dataset policy

This project trains on labeled images only. Images found through Google search
are not automatically genuine, consistently labeled, or licensed for machine
learning, so they must not be scraped directly into `dataset/`.

## Verified public source

MVTec AD is an official industrial inspection dataset published by MVTec:

<https://www.mvtec.com/company/research/datasets/mvtec-ad>

The dataset page states that it is available under **CC BY-NC-SA 4.0**. It is
useful for research and anomaly-detection experiments, but its categories do
not match this application. Do not mix MVTec images into the current folders
or relabel unrelated defects as `Hole`, `Objects`, `Oil Spot`, or `Thread Error`.
Keep it in a separate dataset directory and train a separate model if it is
needed.

## Current model labels

The production model intentionally uses the exact folder labels from the
provided data:

```text
good/         -> Good
hole/         -> Hole
objects/      -> Objects
oil spot/     -> Oil Spot
thread error/ -> Thread Error
```

Only datasets with an explicit license and compatible labels should be added to
these folders. Record the source URL and license alongside any imported data.
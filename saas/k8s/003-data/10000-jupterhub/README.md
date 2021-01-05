Add the JupyterHub repository to Helm and update.
```
$ helm repo add jupyterhub \
   https://jupyterhub.github.io/helm-chart/
$ helm repo update
```
Install (or upgrade/update) the JupyterHub Helm package.
```
$ helm upgrade --install lab-hub jupyterhub/jupyterhub \
  --namespace="data" \
  --version="0.9-dcde99a" \
  --values="values.yml"
```

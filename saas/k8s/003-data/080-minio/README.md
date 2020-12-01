MinIO Operator
1.0.3 provided by MinIO, Inc.
Install on Kubernetes

    Install Operator Lifecycle Manager (OLM), a tool to help manage the Operators running on your cluster.
    $ curl -sL https://github.com/operator-framework/operator-lifecycle-manager/releases/download/v0.17.0/install.sh | bash -s v0.17.0

    Install the operator by running the following command:
    $ kubectl create -f https://operatorhub.io/install/minio-operator.yaml

    This Operator will be installed in the "my-minio-operator" namespace and will be usable from this namespace only.

    After install, watch your operator come up using next command.
    $ kubectl get csv -n my-minio-operator

    To use it, checkout the custom resource definitions (CRDs) introduced by this operator to start using it.

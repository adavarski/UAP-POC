Install/configure awscli:

```
apt install -y python3-pip
pip3 install awscli
aws configure (or export AWS_ACCESS_KEY_ID= ; export AWS_SECRET_ACCESS_KEY=; export AWS_DEFAULT_REGION=us-east-1)
```

Created a new IAM user called k8s-saas with a new group (k8s-saas) that has access to AmazonEC2FullAccess, IAMFullAccess, AmazonS3FullAccess, and AmazonVPCFullAccess (via AWS console).

```
# IAM:ARN
AmazonEC2FullAccess
AmazonRoute53FullAccess
AmazonS3FullAccess
IAMFullAccess
AmazonVPCFullAccess
```

Create KOPS user using aws cli (example):
```

aws iam create-group --group-name k8s-saas
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name k8s-saas
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonRoute53FullAccess --group-name k8s-saas
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --group-name k8s-saas
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/IAMFullAccess --group-name k8s-saas
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess --group-name k8s-saas
aws iam create-user --user-name k8s-saas
aws iam add-user-to-group --user-name k8s-saas --group-name k8s-saas
aws iam create-access-key --user-name k8s-saas

#Record the SecretAccessKey and AccessKeyID in the returned JSON output.
```

Grab access and secrets keys (from previous JSON output for example or from downloaded file `$ cat new_user_credentials.csv` if we use AWS console) and configure your aws profile:k8s-saas  with: 
```
aws configure (or edit ~/.aws/credentials & ~/.aws/config files)
```
Example: Check those creds at:
```
$ cat ~/.aws/credentials 
[default]
aws_access_key_id = 
aws_secret_access_key = 
[k8s-saas]
aws_access_key_id = 
aws_secret_access_key = 

$ cat ~/.aws/config 
cat: cat: No such file or directory
[default]
region = us-east-2
[k8s-saas]
region = us-east-1
```
Next, let's create the s3 bucket with versioning that we need for our kube state (us-east-1). I will call it `k8s-saas-kops-state-dev`. Make sure you turn on versioning.

Create S3 bucket using aws cli(example):

```
aws s3api create-bucket --bucket k8s-saas-kops-state-dev --region us-east-1 
aws s3api put-bucket-versioning --bucket k8s-saas-kops-state-dev --versioning-configuration Status=Enabled

```
Now let's work on the KOPS and Kubernetes segment.

to install kops on linux:
```
curl -LO https://github.com/kubernetes/kops/releases/download/$(curl -s https://api.github.com/repos/kubernetes/kops/releases/latest | grep tag_name | cut -d '"' -f 4)/kops-linux-amd64
chmod +x kops-linux-amd64
sudo mv kops-linux-amd64 /usr/local/bin/kops
```
For other operating systems see their documentation.

Note: What is KOPS? We like to think of it as kubectl for clusters. kops helps you create, destroy, upgrade and maintain production-grade, highly available, Kubernetes clusters from the command line. AWS (Amazon Web Services) is currently officially supported, with GCE and OpenStack in beta support, and VMware vSphere in alpha, and other platforms planned.

KOPS Features

    - Automates the provisioning of Kubernetes clusters in AWS and GCE
    - Deploys Highly Available (HA) Kubernetes Masters
    - Built on a state-sync model for dry-runs and automatic idempotency
    - Ability to generate Terraform
    - Supports managed kubernetes add-ons
    - Command line autocompletion
    - YAML Manifest Based API Configuration
    - Templating and dry-run modes for creating Manifests
    - Choose from eight different CNI Networking providers out-of-the-box
    - Supports upgrading from kube-up
    - Capability to add containers, as hooks, and files to nodes via a cluster manifest



then install kubectl:
```
curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
```
For more information or installation methods, see the docs.

Configuring and Provisioning our Cluster

Now the we have installed the requirements, let's export our variables and provision our cluster.

We need our AWS profile, keys, and S3 bucket name that we created. Export the following:
```
export AWS_PROFILE="k8s-saas"
export AWS_ACCESS_KEY_ID=$(aws configure get k8s-saas.aws_access_key_id)
export AWS_SECRET_ACCESS_KEY=$(aws configure get k8s-saas.aws_secret_access_key)
export KOPS_STATE_STORE=s3://k8s-saas-kops-state-dev
export KUBECONFIG=~/.kube/k8s-saas-AWS-KOPS
```
Note:  KOPS will save your configuration to KUBECONFIG=~/.kube/k8s-saas-AWS-KOPS

Create a new ssh key called k8s-saas with `ssh-keygen`:
```
$ ls  ~/.ssh/k8s-saas*
-rw------- 1 davar davar 1679 Jan 16 09:53 /home/davar/.ssh/k8s-saas
-rw-r--r-- 1 davar davar  394 Jan 16 09:53 /home/davar/.ssh/k8s-saas.pub
```

Once we have done that, let's run kops in the command line to create a master and 3 nodes.(I named mine saas.k8s.local):

Create k8s cluster with kops:

```
kops create cluster \
             --name="saas.k8s.local" \
             --zones="us-east-1a" \
             --master-size="t2.micro" \
             --node-size="t2.micro" \
             --node-count="3" \
             --ssh-public-key="~/.ssh/k8s-saas.pub"
```

Warning: Yes this is free tier, but make sure you setup billing alerts and remember to tear down your cluster when you are not practicing. This will absolutely burn through the free tier compute limit after a day or two of running

Example Outout:

```
...
$ kops create cluster \
>              --name="saas.k8s.local" \
>              --zones="us-east-1a" \
>              --master-size="t2.micro" \
>              --node-size="t2.micro" \
>              --node-count="3" \
>              --ssh-public-key="~/.ssh/k8s-saas.pub"
I0116 10:03:44.338195   21095 create_cluster.go:1537] Using SSH public key: /home/davar/.ssh/k8s-saas.pub
I0116 10:03:48.729134   21095 create_cluster.go:555] Inferred --cloud=aws from zone "us-east-1a"
I0116 10:03:50.374124   21095 subnets.go:184] Assigned CIDR 172.20.32.0/19 to subnet us-east-1a
Previewing changes that will be made:

...

Must specify --yes to apply changes

Cluster configuration has been created.

Suggestions:
 * list clusters with: kops get cluster
 * edit this cluster with: kops edit cluster saas.k8s.local
 * edit your node instance group: kops edit ig --name=saas.k8s.local nodes
 * edit your master instance group: kops edit ig --name=saas.k8s.local master-us-east-1a

Finally configure your cluster with: kops update cluster --name saas.k8s.local --yes
```
Check:

```
$ kops get cluster
NAME		CLOUD	ZONES
saas.k8s.local	aws	us-east-1a
```

All configuration and your cluster state is stored in the S3 bucket that you created:

```
$ aws s3 ls s3://k8s-saas-kops-state-dev/saas.k8s.local/
                           PRE instancegroup/
                           PRE pki/
2021-01-16 10:03:53       5191 cluster.spec
2021-01-16 10:03:52       1101 config
```
Create k8s cluster with kops (Final step: --yes):

```
kops update cluster --name saas.k8s.local --yes
```

Example output:
```
$ kops update cluster --name saas.k8s.local --yes
I0116 11:04:27.580148    9325 apply_cluster.go:545] Gossip DNS: skipping DNS validation
I0116 11:04:30.349723    9325 executor.go:103] Tasks: 0 done / 92 total; 42 can run
I0116 11:04:31.770265    9325 vfs_castore.go:590] Issuing new certificate: "ca"
I0116 11:04:31.789921    9325 vfs_castore.go:590] Issuing new certificate: "etcd-manager-ca-events"
I0116 11:04:31.940586    9325 vfs_castore.go:590] Issuing new certificate: "etcd-clients-ca"
I0116 11:04:32.034240    9325 vfs_castore.go:590] Issuing new certificate: "apiserver-aggregator-ca"
I0116 11:04:32.082461    9325 vfs_castore.go:590] Issuing new certificate: "etcd-manager-ca-main"
I0116 11:04:32.110194    9325 vfs_castore.go:590] Issuing new certificate: "etcd-peers-ca-main"
I0116 11:04:32.193391    9325 vfs_castore.go:590] Issuing new certificate: "etcd-peers-ca-events"
I0116 11:04:35.874394    9325 executor.go:103] Tasks: 42 done / 92 total; 26 can run
I0116 11:04:37.021057    9325 vfs_castore.go:590] Issuing new certificate: "kube-controller-manager"
I0116 11:04:37.069603    9325 vfs_castore.go:590] Issuing new certificate: "apiserver-aggregator"
I0116 11:04:37.082438    9325 vfs_castore.go:590] Issuing new certificate: "kubelet"
I0116 11:04:37.161231    9325 vfs_castore.go:590] Issuing new certificate: "kubecfg"
I0116 11:04:37.228699    9325 vfs_castore.go:590] Issuing new certificate: "apiserver-proxy-client"
I0116 11:04:37.234030    9325 vfs_castore.go:590] Issuing new certificate: "kubelet-api"
I0116 11:04:37.268865    9325 vfs_castore.go:590] Issuing new certificate: "kops"
I0116 11:04:37.346481    9325 vfs_castore.go:590] Issuing new certificate: "kube-scheduler"
I0116 11:04:37.683070    9325 vfs_castore.go:590] Issuing new certificate: "kube-proxy"
I0116 11:04:40.278350    9325 executor.go:103] Tasks: 68 done / 92 total; 20 can run
I0116 11:04:43.554976    9325 executor.go:103] Tasks: 88 done / 92 total; 3 can run
I0116 11:04:44.718474    9325 vfs_castore.go:590] Issuing new certificate: "master"
I0116 11:04:46.729677    9325 executor.go:103] Tasks: 91 done / 92 total; 1 can run
W0116 11:04:47.343917    9325 executor.go:128] error running task "LoadBalancerAttachment/api-master-us-east-1a" (9m59s remaining to succeed): error attaching autoscaling group to ELB: ValidationError: Provided Load Balancers may not be valid. Please ensure they exist and try again.
	status code: 400, request id: 609a8530-c015-414c-bd47-34f765872dec
I0116 11:04:47.343980    9325 executor.go:143] No progress made, sleeping before retrying 1 failed task(s)
I0116 11:04:57.344178    9325 executor.go:103] Tasks: 91 done / 92 total; 1 can run
I0116 11:04:59.119563    9325 executor.go:103] Tasks: 92 done / 92 total; 0 can run
I0116 11:05:01.373980    9325 update_cluster.go:308] Exporting kubecfg for cluster
kops has set your kubectl context to saas.k8s.local

Cluster is starting.  It should be ready in a few minutes.

Suggestions:
 * validate cluster: kops validate cluster --wait 10m
 * list nodes: kubectl get nodes --show-labels
 * ssh to the master: ssh -i ~/.ssh/id_rsa ubuntu@api.saas.k8s.local
 * the ubuntu user is specific to Ubuntu. If not using Ubuntu please use the appropriate user based on your OS.
 * read about installing addons at: https://kops.sigs.k8s.io/operations/addons.
```

If you get any errors, try running:
```
kops update cluster --name saas.k8s.local
```
Example output:
```
$ kops update cluster --name saas.k8s.local
I0116 11:08:48.837743   15693 apply_cluster.go:545] Gossip DNS: skipping DNS validation
I0116 11:08:48.880985   15693 executor.go:103] Tasks: 0 done / 92 total; 42 can run
I0116 11:08:50.016114   15693 executor.go:103] Tasks: 42 done / 92 total; 26 can run
I0116 11:08:50.911431   15693 executor.go:103] Tasks: 68 done / 92 total; 20 can run
I0116 11:08:52.119690   15693 executor.go:103] Tasks: 88 done / 92 total; 3 can run
I0116 11:08:53.210878   15693 executor.go:103] Tasks: 91 done / 92 total; 1 can run
I0116 11:08:53.517883   15693 executor.go:103] Tasks: 92 done / 92 total; 0 can run
No changes need to be applied

```

and to validate:
```
kops validate cluster --name saas.k8s.local
```
Example output:
```
$ kops validate cluster --wait 10m
Using cluster from kubectl context: saas.k8s.local

Validating cluster saas.k8s.local

INSTANCE GROUPS
NAME			ROLE	MACHINETYPE	MIN	MAX	SUBNETS
master-us-east-1a	Master	t2.micro	1	1	us-east-1a
nodes			Node	t2.micro	3	3	us-east-1a

NODE STATUS
NAME				ROLE	READY
ip-172-20-40-101.ec2.internal	node	True
ip-172-20-45-14.ec2.internal	node	True
ip-172-20-46-229.ec2.internal	node	True
ip-172-20-59-202.ec2.internal	master	True

Your cluster saas.k8s.local is ready

```

Once this cluster is complete, you should be able to see it in your EC2 Dashboard. It will also save your configuration to .kube/ (~/.kube/k8s-saas-AWS-KOPS) in you home directory.

Try:
```
kubectl get nodes
```
Example output:
```
$ kubectl get nodes --show-labels
NAME                            STATUS   ROLES    AGE     VERSION    LABELS
ip-172-20-40-101.ec2.internal   Ready    node     67s     v1.18.14   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/instance-type=t2.micro,beta.kubernetes.io/os=linux,failure-domain.beta.kubernetes.io/region=us-east-1,failure-domain.beta.kubernetes.io/zone=us-east-1a,kops.k8s.io/instancegroup=nodes,kubernetes.io/arch=amd64,kubernetes.io/hostname=ip-172-20-40-101.ec2.internal,kubernetes.io/os=linux,kubernetes.io/role=node,node-role.kubernetes.io/node=,node.kubernetes.io/instance-type=t2.micro,topology.kubernetes.io/region=us-east-1,topology.kubernetes.io/zone=us-east-1a
ip-172-20-45-14.ec2.internal    Ready    node     69s     v1.18.14   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/instance-type=t2.micro,beta.kubernetes.io/os=linux,failure-domain.beta.kubernetes.io/region=us-east-1,failure-domain.beta.kubernetes.io/zone=us-east-1a,kops.k8s.io/instancegroup=nodes,kubernetes.io/arch=amd64,kubernetes.io/hostname=ip-172-20-45-14.ec2.internal,kubernetes.io/os=linux,kubernetes.io/role=node,node-role.kubernetes.io/node=,node.kubernetes.io/instance-type=t2.micro,topology.kubernetes.io/region=us-east-1,topology.kubernetes.io/zone=us-east-1a
ip-172-20-46-229.ec2.internal   Ready    node     69s     v1.18.14   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/instance-type=t2.micro,beta.kubernetes.io/os=linux,failure-domain.beta.kubernetes.io/region=us-east-1,failure-domain.beta.kubernetes.io/zone=us-east-1a,kops.k8s.io/instancegroup=nodes,kubernetes.io/arch=amd64,kubernetes.io/hostname=ip-172-20-46-229.ec2.internal,kubernetes.io/os=linux,kubernetes.io/role=node,node-role.kubernetes.io/node=,node.kubernetes.io/instance-type=t2.micro,topology.kubernetes.io/region=us-east-1,topology.kubernetes.io/zone=us-east-1a
ip-172-20-59-202.ec2.internal   Ready    master   2m20s   v1.18.14   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/instance-type=t2.micro,beta.kubernetes.io/os=linux,failure-domain.beta.kubernetes.io/region=us-east-1,failure-domain.beta.kubernetes.io/zone=us-east-1a,kops.k8s.io/instancegroup=master-us-east-1a,kubernetes.io/arch=amd64,kubernetes.io/hostname=ip-172-20-59-202.ec2.internal,kubernetes.io/os=linux,kubernetes.io/role=master,node-role.kubernetes.io/master=,node.kubernetes.io/instance-type=t2.micro,topology.kubernetes.io/region=us-east-1,topology.kubernetes.io/zone=us-east-1a
```

You should be able to see the nodes you just created.

You can now use this cluster to try things out, but again, i'd recommend k3s/minikube for testing. This is a good piece of knowledge to have in your professional tool belt.

Check kubectl config file generated by KOPS:

```
$ cat ~/.kube/k8s-saas-AWS-KOPS 
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJkekNDQVIyZ0F3SUJBZ0lCQURBS0JnZ3Foa2pPUFFRREFqQWpNU0V3SHdZRFZRUUREQmhyTTNNdGMyVnkKZG1WeUxXTmhRREUyTURVNE5USXhOemN3SGhjTk1qQXhNVEl3TURZd01qVTNXaGNOTXpBeE1URTRNRFl3TWpVMwpXakFqTVNFd0h3WURWUVFEREJock0zTXRjMlZ5ZG1WeUxXTmhRREUyTURVNE5USXhOemN3V1RBVEJnY3Foa2pPClBRSUJCZ2dxaGtqT1BRTUJCd05DQUFSMkNMVzZkSEs0UnVmLzJRaWNvd0JtNFJ6SVJJK3lTSlZ6TlhWSHRtR1YKbzRFWml3Z3JOQnBzK2tlNkhscUJlSzVWQlArTjVHYmp2SE9LZUM2U3NJOFJvMEl3UURBT0JnTlZIUThCQWY4RQpCQU1DQXFRd0R3WURWUjBUQVFIL0JBVXdBd0VCL3pBZEJnTlZIUTRFRmdRVWgvdkUwQVlXa3ZqTGdXcUYwZWNTCjZOZEp1SjB3Q2dZSUtvWkl6ajBFQXdJRFNBQXdSUUloQUpmK0J6d2JTaDBiaTdldEpIQlppRzdXaG9PaUNHaEUKcVMzTUhXQXhsWm1GQWlBSDIwYzhPUnFkbElSYnJ0Y0paMlpHdGw1Wk5aMnNNUW9LNENoa01IQlBoUT09Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
    server: https://192.168.0.100:6443
  name: default
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUMwekNDQWJ1Z0F3SUJBZ0lNRmxxckNQbGZnTFVseFp0OU1BMEdDU3FHU0liM0RRRUJDd1VBTUJVeEV6QVIKQmdOVkJBTVRDbXQxWW1WeWJtVjBaWE13SGhjTk1qRXdNVEUwTURrd05ETXhXaGNOTXpFd01URTBNRGt3TkRNeApXakFWTVJNd0VRWURWUVFERXdwcmRXSmxjbTVsZEdWek1JSUJJakFOQmdrcWhraUc5dzBCQVFFRkFBT0NBUThBCk1JSUJDZ0tDQVFFQXBxTXBobjlzZk1aMEZuYzJTSjVPc290YWMzNmVMNDRZTGxQRmFiZzlYY1B1bFZ5RGFDNDEKUU4ybUgyTnBNSDdOVkZDUE1CU1pBUkRzT0RTaExqV2VmRUFXZGxsNjlsdkIzSTE1UFdmNGczb1kvYVlBNC9lMgpqelJCN0ZBUjBva1RIdE0raEZCYnh0LzFSMWRaNEI0Q0tkZ2FLZ2psanBpWmFmNXEraWo4U3EzTjdVcENDc3ExClM0d05mbTN2ZXA3RG5TSWRVOVI5eVlkRGFCbjJBNjJKTTRoTXFML2kzeTNWNUxnTUhqWGtkaFJnMlFsaGhJcVIKQmFxdDZYRDFkUXRSUWhRMXFuYnJ4RmI1ZTljM2ZuV1ZHZ3p4QUloV1VadW01a3FTdXJQbEw2dEJBOE5iWnU5egpoc0tkQUZhb1dDcmVKWFQyNVFOOGJJRzZBVTE2QWgrMm9RSURBUUFCb3lNd0lUQU9CZ05WSFE4QkFmOEVCQU1DCkFRWXdEd1lEVlIwVEFRSC9CQVV3QXdFQi96QU5CZ2txaGtpRzl3MEJBUXNGQUFPQ0FRRUFUbzBSaGwvUkVhTlIKTVhaeEZ4L1VwczdXVjFFc3FkVDE0Nlo2QXJmU0VPRlVWUzBlWVBWWFh6aG9ZQVlZbE9VMEdMQk94NjZqTHVOUgpCaVlGYVZHWldmQVBiV3F0VXh0V25iSEc0Mi8rWEtsWmFxbWxCNFYwamZXUmpkckFvMWd4YTBndnQ4OXorV042Ckh2TWJoM0pNTlFucWw1dDArR0xkY2Z4NDB5QmdwOEc3OXFWYktvQXV5Y2wvSHVjYXFGY1hHMVhSNE1aZzJNQTQKQUFpbjN0QjBWNTJ0TlVUNndpdWxiaGltRUtJaWt1WDF0ajdDS2NaYkNUVlhUYTFObDhSY3B0dTc0cGhybHZoRgpiUjllZUxpbDY2TjVNS2JXSEsvZTJzcFFraktNYytYV1huVGdYN0E4VUNlWWtnRWhZRld5aVdzNnJXRGlGWG1ICnVadXlxQnE4NlE9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg==
    server: https://api-saas-k8s-local-c3t1l2-30550779.us-east-1.elb.amazonaws.com
  name: saas.k8s.local
contexts:
- context:
    cluster: default
    user: default
  name: default
- context:
    cluster: saas.k8s.local
    user: saas.k8s.local
  name: saas.k8s.local
current-context: saas.k8s.local
kind: Config
preferences: {}
users:
- name: default
  user:
    client-certificate-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJrVENDQVRlZ0F3SUJBZ0lJQXRpNG1oWjFQTll3Q2dZSUtvWkl6ajBFQXdJd0l6RWhNQjhHQTFVRUF3d1kKYXpOekxXTnNhV1Z1ZEMxallVQXhOakExT0RVeU1UYzNNQjRYRFRJd01URXlNREEyTURJMU4xb1hEVEl4TVRFeQpNREEyTURJMU4xb3dNREVYTUJVR0ExVUVDaE1PYzNsemRHVnRPbTFoYzNSbGNuTXhGVEFUQmdOVkJBTVRESE41CmMzUmxiVHBoWkcxcGJqQlpNQk1HQnlxR1NNNDlBZ0VHQ0NxR1NNNDlBd0VIQTBJQUJFUjNMc3hxSmh3RU9qQVgKcmFuanlvVHpRYVpEVVYzbzRFOGhvVTBrU2JxUDR2blRWV0tCMnNpNTN0YW96enlBWENnR0VoUW1aNm5mMHcyMgpEbTFHQ3JpalNEQkdNQTRHQTFVZER3RUIvd1FFQXdJRm9EQVRCZ05WSFNVRUREQUtCZ2dyQmdFRkJRY0RBakFmCkJnTlZIU01FR0RBV2dCVEVHWnROZS96WlpEQVlPS2loSDM0VGFEVGgzekFLQmdncWhrak9QUVFEQWdOSUFEQkYKQWlFQWdrOGFWbGFuMlEzNVhKNzdnd1NPNllmZ2g0SWJlSGtleVJBSHRyallSNE1DSUJ5ZE41WEswRS82WHM3cwppSGcwdHllU0hwbTUyV1FFeXFsOEV2QXNESFEvCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0KLS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJkekNDQVIyZ0F3SUJBZ0lCQURBS0JnZ3Foa2pPUFFRREFqQWpNU0V3SHdZRFZRUUREQmhyTTNNdFkyeHAKWlc1MExXTmhRREUyTURVNE5USXhOemN3SGhjTk1qQXhNVEl3TURZd01qVTNXaGNOTXpBeE1URTRNRFl3TWpVMwpXakFqTVNFd0h3WURWUVFEREJock0zTXRZMnhwWlc1MExXTmhRREUyTURVNE5USXhOemN3V1RBVEJnY3Foa2pPClBRSUJCZ2dxaGtqT1BRTUJCd05DQUFRMVMzeGNrb1g5dFhGVkx3djg5OG9YZ2dQVmFLakVGQldQMnFENDZVblMKTjc0bEpOTjJGQyt6cUJqdEdQQzBNYVFReUhvU0lPR3hqa0RtRzc3b25hTENvMEl3UURBT0JnTlZIUThCQWY4RQpCQU1DQXFRd0R3WURWUjBUQVFIL0JBVXdBd0VCL3pBZEJnTlZIUTRFRmdRVXhCbWJUWHY4MldRd0dEaW9vUjkrCkUyZzA0ZDh3Q2dZSUtvWkl6ajBFQXdJRFNBQXdSUUlnTVJMTHlKaHVybW5iUmxqUjZjMUZIWGRoby81SlJoQ1QKVWloVGViRXhGaTBDSVFEelNCQXJwMUEzOHFubGRybXNtM0o2c0tOL1RuT0xvaEkwUEE0RnNiSXZFZz09Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
    client-key-data: LS0tLS1CRUdJTiBFQyBQUklWQVRFIEtFWS0tLS0tCk1IY0NBUUVFSVB1WVJpZGJNMWFLT0pCRFNITmoyQytVSmVLNTk0bTNMcDNJTXVGbEk5dm9vQW9HQ0NxR1NNNDkKQXdFSG9VUURRZ0FFUkhjdXpHb21IQVE2TUJldHFlUEtoUE5CcGtOUlhlamdUeUdoVFNSSnVvL2krZE5WWW9IYQp5TG5lMXFqUFBJQmNLQVlTRkNabnFkL1REYllPYlVZS3VBPT0KLS0tLS1FTkQgRUMgUFJJVkFURSBLRVktLS0tLQo=
- name: saas.k8s.local
  user:
    client-certificate-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUMrekNDQWVPZ0F3SUJBZ0lNRmxxckNqcXpMMmFtVmUxY01BMEdDU3FHU0liM0RRRUJDd1VBTUJVeEV6QVIKQmdOVkJBTVRDbXQxWW1WeWJtVjBaWE13SGhjTk1qRXdNVEUwTURrd05ETTNXaGNOTXpFd01URTBNRGt3TkRNMwpXakFyTVJjd0ZRWURWUVFLRXc1emVYTjBaVzA2YldGemRHVnljekVRTUE0R0ExVUVBeE1IYTNWaVpXTm1aekNDCkFTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBSzhyd25DQVAxMEJaYU5pQ25zc25saTAKUmtycDMwVVdRN1BXV2wxSWZrTEQ0QW1id1RmMjdVVTRNWFk3Uyt0eU9zRVA5UUMwQzlQUnZJb2I0M3R3QS9CUQp5YUdwdXVuTjAyYjNKS0ZINnNBRzAxZUEvbFNIUkZtRmN1MFhpQUI5anBnbndBVTlmZEtKbDl2K3k5blpDMW5FCkFYRU9iQm5kMjJvUXhxNlkwQVZSUVEwU2lLOGNZaStGRzcxMm5zRDZBN0EyQW9yanFsK1RLWHlDOC96K1RiVWQKQ3MrMXZkSzNpeTFkblNuaGVPUnNiRkQrSUE1VVBUcHlYTzFVeUxwQWxqTkthUG81bXdxOGdkVnd5OG5OWUk2WQpLTC93SXlZbGx1MEY2SEU1QW1RNnFlZ1ZIRjJOMStQVWpVamtNZlpOY3I4UDhkQkZqdDR3NjNMVUttUmtpYWtDCkF3RUFBYU0xTURNd0RnWURWUjBQQVFIL0JBUURBZ2VBTUJNR0ExVWRKUVFNTUFvR0NDc0dBUVVGQndNQ01Bd0cKQTFVZEV3RUIvd1FDTUFBd0RRWUpLb1pJaHZjTkFRRUxCUUFEZ2dFQkFCTW8yWWFsSnpDNHBudGtWU2xna08ycAozNDBuOVZRd2NoNFdybkFPSFBYSy85MC9nOTA2VHZjbHV0RGFmYUdra2liNEZJdURvQlZaVDM1QTFTaE9JMDFPCklBdGNDVjJSWnRrRlhZMzEzeUZDclk5NDRXOVhpR3JMY255dkNpY1ZMSFhBaDJtTUNCRlhWWFFRakRuYjJoeSsKZkFQS1lkT0dJQ2R2eHhDS21zMnNrRXg3ZmorL05EMFloaU5ualpnR2RzekoralI4UUxab21QM0V6ZGYzMDlNVwp3ZTJCb0lZMU5iaFFXWWtmZlM0cE55N1ViOTlmRmMwYk84WmsvanczdWJ3YVlPSmxvcGsrQnViNzV1ZC9tTU0xCkR4d1Ixd3hKcGIvL3NmcTFMWldkOHlrSTAvelNpYlRtQ2J2bnZLNjNhekc3WHVFTURUck0xY1E1b09JMW9KRT0KLS0tLS1FTkQgQ0VSVElGSUNBVEUtLS0tLQo=
    client-key-data: LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcEFJQkFBS0NBUUVBcnl2Q2NJQS9YUUZsbzJJS2V5eWVXTFJHU3VuZlJSWkRzOVphWFVoK1FzUGdDWnZCCk4vYnRSVGd4ZGp0TDYzSTZ3US8xQUxRTDA5RzhpaHZqZTNBRDhGREpvYW02NmMzVFp2Y2tvVWZxd0FiVFY0RCsKVklkRVdZVnk3UmVJQUgyT21DZkFCVDE5MG9tWDIvN0wyZGtMV2NRQmNRNXNHZDNiYWhER3JwalFCVkZCRFJLSQpyeHhpTDRVYnZYYWV3UG9Ec0RZQ2l1T3FYNU1wZklMei9QNU50UjBLejdXOTByZUxMVjJkS2VGNDVHeHNVUDRnCkRsUTlPbkpjN1ZUSXVrQ1dNMHBvK2ptYkNyeUIxWERMeWMxZ2pwZ292L0FqSmlXVzdRWG9jVGtDWkRxcDZCVWMKWFkzWDQ5U05TT1F4OWsxeXZ3L3gwRVdPM2pEcmN0UXFaR1NKcVFJREFRQUJBb0lCQUFkREhDdUgxWmlvZnlzMwozRkZnOXl1Y1JiSERZbU05Mmc2eG0wdDB2dTlMcVpVY1pQVktBbW8xT3krNG81d2VQenRUeXdkOWIyUVh0T1pRCjQ2eHhaMmhtSVFYWHdJMEM0SG5MVXpSd2c5WDBNbnpWTnpZUDZaQ0FqTlF4Zm96dXA2TzlPaU85ZWVMcG0yY2UKZUJzTytYNStOc1VXSVBzdGpIUW9QK1JySjJ5ZzhSdEdrdHU5eE1LcEIySDljRGdlTDVleWZaNndyLzA4Zk1HeAoxeERzM0hnQWZ1VHVpN2cwYXM1V3QwdnBKeVhwYVMvbVZ3ckt0TURkN3A4SEZDL0ovbkNYNzh3M1paK2sxUW11CkhBV2xhZjhrZHZjbmd1amhveE5KNlQ1NDBvZXFKV01WOFdxNFZEVGh5Q0s3ODROSU1sY1ZLS3l3Nm85SVA5N3UKVkNkU3VRVUNnWUVBNk0wbUM3NDFwSWhPU2toaG1hSEJjOVY0bksxVk1iR2JaRUFFdFovTVlWNVk1ZFlkQVU2Uwp6aWFkdXVwQng3RG5lRmVtUU82YUFELzN1aXF1d1VhUXIwbnlmeGpGRVluYzBMZFdCV1pqaTk1YnhEb0ZNMmorClFwSXpFaEVFMVpZeWZXaGJvV0dadUhJZjliMjJ4SE15cTlZM3dmeFF5S0xIV1EyZDZReTdycWNDZ1lFQXdLQnYKMGY1UTRacjhOVlNYcjRRd05CZjUyN2czUUovNmFNbXN6OFVnMVB3ZzNrUytkVGJHVTgzS1lISkZ6SjBXekhkRQpqOGJtZUpFVEltOHlUZ2xMUUhYN0VZaVdBRGYvWVZ1ZURieXh3V292a1lDd3k3V1FmQitQT3Zmazd6dDBKWmxTCldzckxQTkpJbk1WK0R1UkdQNnpmWHR2UXFmSm9uWDhhbys1VTN5OENnWUVBdldFTm55WG51MlhLMG5EWlBNSXkKZmdsZEtZOXpxNDVBeTZlc2JSZUdpbnJXdEhtZWRNNjNiQTVMNTU3alRoQWg3R3JZNng5dWxkOXpwYlZOSGh0SwpYZDQ0NzUxMXl0T3NsdmxkZDkwWUE5TXFNWkhGWTFINDdLekQ4T0Y1WGtybDNkREVJWjhsRHIrQU1rZFhZNjRXCnZXVHhUTlAxMzVGVHU2VHhIWUZuT0NjQ2dZQUpvUHQxOTh4N09wQVk2bDJhdkhUY1pjWnJvVTNCd1QxM04zMlgKOVRhcTd0K0Z2Tkg0Y2dCLzkrMkIzTFI0Z2ZHOXpzaHlsM0sxM0Nxa3NnSkZ2Rmt5bzdNbE1UcXVVQjVIODVoMgovMU96WTJkMVVvV0Q2Vk9Mb05nOERtQTNSVFdORzVqdVNPelg0WHNYdXhlT0R3TWo4N1JHdlo0MG9KVnlLZ1JlCjVXOEFxUUtCZ1FDdGhlKzJ2ekdnUDkrdmtaM2ozNE9kNUY5VmxXd2ZtSk0zZHR4T1NESG9ZNTFnN25aRWhiY1cKWm5EOHY0VHE1cnA4RFhBWTZ0MUQxeVN6bGVGMVhJSFA3b1NRblFVREN6cXhzdzZWVjMycnFOSitzVkNLYWZiNApZRk9qNC80amcwbDFLMUY1QXVtUzdDeDZPbjVxYUVvODZUODRWM3h0cjVOeU45blM4NFVpWXc9PQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo=
```
Check k8s cluster:
```
$ kubectl cluster-info
Kubernetes master is running at https://api-saas-k8s-local-c3t1l2-30550779.us-east-1.elb.amazonaws.com
KubeDNS is running at https://api-saas-k8s-local-c3t1l2-30550779.us-east-1.elb.amazonaws.com/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.

$ kubectl get all --all-namespaces
NAMESPACE     NAME                                                        READY   STATUS             RESTARTS   AGE
kube-system   pod/dns-controller-c4dc48d8d-pz9v6                          1/1     Running            0          35m
kube-system   pod/etcd-manager-events-ip-172-20-59-202.ec2.internal       1/1     Running            0          35m
kube-system   pod/etcd-manager-main-ip-172-20-59-202.ec2.internal         1/1     Running            0          35m
kube-system   pod/kops-controller-lzbtn                                   1/1     Running            3          35m
kube-system   pod/kube-apiserver-ip-172-20-59-202.ec2.internal            2/2     Running            3          35m
kube-system   pod/kube-controller-manager-ip-172-20-59-202.ec2.internal   1/1     Running            2          35m
kube-system   pod/kube-dns-6c699b5445-ln68m                               3/3     Running            2          34m
kube-system   pod/kube-dns-6c699b5445-wdxkw                               3/3     Running            2          35m
kube-system   pod/kube-dns-autoscaler-cd7778b7b-hcjz4                     1/1     Running            0          35m
kube-system   pod/kube-proxy-ip-172-20-40-101.ec2.internal                1/1     Running            0          33m
kube-system   pod/kube-proxy-ip-172-20-45-14.ec2.internal                 1/1     Running            0          33m
kube-system   pod/kube-proxy-ip-172-20-46-229.ec2.internal                1/1     Running            0          34m
kube-system   pod/kube-proxy-ip-172-20-59-202.ec2.internal                1/1     Running            0          35m
kube-system   pod/kube-scheduler-ip-172-20-59-202.ec2.internal            1/1     Running            1          35m

NAMESPACE     NAME                 TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)         AGE
default       service/kubernetes   ClusterIP   100.64.0.1    <none>        443/TCP         36m
kube-system   service/kube-dns     ClusterIP   100.64.0.10   <none>        53/UDP,53/TCP   36m

NAMESPACE     NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                     AGE
kube-system   daemonset.apps/kops-controller   1         1         1       1            1           node-role.kubernetes.io/master=   36m

NAMESPACE     NAME                                  READY   UP-TO-DATE   AVAILABLE   AGE
kube-system   deployment.apps/dns-controller        1/1     1            1           36m
kube-system   deployment.apps/kube-dns              2/2     2            2           36m
kube-system   deployment.apps/kube-dns-autoscaler   1/1     1            1           36m

NAMESPACE     NAME                                            DESIRED   CURRENT   READY   AGE
kube-system   replicaset.apps/dns-controller-c4dc48d8d        1         1         1       36m
kube-system   replicaset.apps/kube-dns-6c699b5445             2         2         2       36m
kube-system   replicaset.apps/kube-dns-autoscaler-cd7778b7b   1         1         1       36m
```


Login to master node: 

```
aws ec2 describe-instances  --filter 'Name=instance-state-name,Values=running' |   jq -r '.Reservations[].Instances[] | [.InstanceId, .PublicDnsName, .Tags[].Value] | @json'

$ ssh -i ~/.ssh/k8s-saas ubuntu@ec2-3-238-140-116.compute-1.amazonaws.com
The authenticity of host 'ec2-3-238-140-116.compute-1.amazonaws.com (3.238.140.116)' can't be established.
ECDSA key fingerprint is SHA256:GGIw6WFWDyxJtdkh3sx1RxiyDVziaNRkNuB0pEKVirg.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'ec2-3-238-140-116.compute-1.amazonaws.com,3.238.140.116' (ECDSA) to the list of known hosts.
Welcome to Ubuntu 20.04.1 LTS (GNU/Linux 5.4.0-1029-aws x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of Sat Jan 16 09:18:21 UTC 2021

  System load:  0.07              Processes:                150
  Usage of /:   6.4% of 61.98GB   Users logged in:          0
  Memory usage: 78%               IPv4 address for docker0: 172.17.0.1
  Swap usage:   0%                IPv4 address for eth0:    172.20.59.202

58 updates can be installed immediately.
27 of these updates are security updates.
To see these additional updates run: apt list --upgradable



The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.

To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.


ubuntu@ip-172-20-59-202:~$ sudo su -
root@ip-172-20-59-202:~# docker ps -a
CONTAINER ID        IMAGE                                COMMAND                  CREATED             STATUS                     PORTS               NAMES
b81ad0a9a12e        kope/kops-controller                 "/usr/bin/kops-contr…"   7 minutes ago       Up 7 minutes                                   k8s_kops-controller_kops-controller-lzbtn_kube-system_ec8d53c9-8a58-4566-aa26-38c7071ce269_0
da2edb057c83        kope/dns-controller                  "/usr/bin/dns-contro…"   7 minutes ago       Up 7 minutes                                   k8s_dns-controller_dns-controller-c4dc48d8d-pz9v6_kube-system_3764535b-fca0-4225-853e-291cb062761e_0
53f7a900263d        k8s.gcr.io/pause-amd64:3.2           "/pause"                 8 minutes ago       Up 7 minutes                                   k8s_POD_kops-controller-lzbtn_kube-system_ec8d53c9-8a58-4566-aa26-38c7071ce269_0
a024cabce0fd        k8s.gcr.io/pause-amd64:3.2           "/pause"                 8 minutes ago       Up 7 minutes                                   k8s_POD_dns-controller-c4dc48d8d-pz9v6_kube-system_3764535b-fca0-4225-853e-291cb062761e_0
df710b7937f0        f17e261f4c8a                         "/usr/local/bin/kube…"   9 minutes ago       Up 9 minutes                                   k8s_kube-apiserver_kube-apiserver-ip-172-20-59-202.ec2.internal_kube-system_91672a4c98cc647c69a20486df68cfd1_1
6dab29f0116a        kope/kube-apiserver-healthcheck      "/usr/bin/kube-apise…"   9 minutes ago       Up 9 minutes                                   k8s_healthcheck_kube-apiserver-ip-172-20-59-202.ec2.internal_kube-system_91672a4c98cc647c69a20486df68cfd1_0
7bfd9f1f32d3        k8s.gcr.io/kube-proxy                "/usr/local/bin/kube…"   9 minutes ago       Up 9 minutes                                   k8s_kube-proxy_kube-proxy-ip-172-20-59-202.ec2.internal_kube-system_7bfb5c367b763ec8c6e1ffc7d783f87c_0
acf332358d61        k8s.gcr.io/kube-scheduler            "/usr/local/bin/kube…"   9 minutes ago       Up 9 minutes                                   k8s_kube-scheduler_kube-scheduler-ip-172-20-59-202.ec2.internal_kube-system_f59f2e6f857285c3cd3a7e1d00cefdca_0
71524e492e7e        k8s.gcr.io/kube-controller-manager   "/usr/local/bin/kube…"   9 minutes ago       Up 9 minutes                                   k8s_kube-controller-manager_kube-controller-manager-ip-172-20-59-202.ec2.internal_kube-system_9b411dbf5fbe9944149a8c389b7a7285_0
58694e3190c5        kopeio/etcd-manager                  "/bin/sh -c 'mkfifo …"   9 minutes ago       Up 9 minutes                                   k8s_etcd-manager_etcd-manager-events-ip-172-20-59-202.ec2.internal_kube-system_2ea2eb3d7b58dacbaedf563a925fc6a1_0
211822e2f740        kopeio/etcd-manager                  "/bin/sh -c 'mkfifo …"   9 minutes ago       Up 9 minutes                                   k8s_etcd-manager_etcd-manager-main-ip-172-20-59-202.ec2.internal_kube-system_159b1da5b608410bd18da2b9ae8dfbc9_0
7da5ffc57c5b        k8s.gcr.io/kube-apiserver            "/usr/local/bin/kube…"   10 minutes ago      Exited (2) 9 minutes ago                       k8s_kube-apiserver_kube-apiserver-ip-172-20-59-202.ec2.internal_kube-system_91672a4c98cc647c69a20486df68cfd1_0
b9796e0540dc        k8s.gcr.io/pause-amd64:3.2           "/pause"                 10 minutes ago      Up 10 minutes                                  k8s_POD_kube-proxy-ip-172-20-59-202.ec2.internal_kube-system_7bfb5c367b763ec8c6e1ffc7d783f87c_0
ff27a1bc104d        k8s.gcr.io/pause-amd64:3.2           "/pause"                 10 minutes ago      Up 10 minutes                                  k8s_POD_kube-scheduler-ip-172-20-59-202.ec2.internal_kube-system_f59f2e6f857285c3cd3a7e1d00cefdca_0
9fa1369318b6        k8s.gcr.io/pause-amd64:3.2           "/pause"                 10 minutes ago      Up 10 minutes                                  k8s_POD_etcd-manager-main-ip-172-20-59-202.ec2.internal_kube-system_159b1da5b608410bd18da2b9ae8dfbc9_0
fc0289cd5d70        k8s.gcr.io/pause-amd64:3.2           "/pause"                 10 minutes ago      Up 10 minutes                                  k8s_POD_kube-apiserver-ip-172-20-59-202.ec2.internal_kube-system_91672a4c98cc647c69a20486df68cfd1_0
3491b73b93da        k8s.gcr.io/pause-amd64:3.2           "/pause"                 10 minutes ago      Up 10 minutes                                  k8s_POD_etcd-manager-events-ip-172-20-59-202.ec2.internal_kube-system_2ea2eb3d7b58dacbaedf563a925fc6a1_0
9a283d467a80        k8s.gcr.io/pause-amd64:3.2           "/pause"                 10 minutes ago      Up 10 minutes                                  k8s_POD_kube-controller-manager-ip-172-20-59-202.ec2.internal_kube-system_9b411dbf5fbe9944149a8c389b7a7285_0
2d4485957bd6        protokube:1.18.2                     "/usr/bin/protokube …"   10 minutes ago      Up 10 minutes                                  protokube
```


When you are finished, go ahead and bring it down to save your free tier compute hours:

```
kops delete cluster saas.k8s.local --yes
```
Example output:
```
$ kops delete cluster saas.k8s.local --yes
TYPE			NAME										ID
autoscaling-config	master-us-east-1a.masters.saas.k8s.local-20210116090440				master-us-east-1a.masters.saas.k8s.local-20210116090440
autoscaling-config	nodes.saas.k8s.local-20210116090440						nodes.saas.k8s.local-20210116090440
autoscaling-group	master-us-east-1a.masters.saas.k8s.local					master-us-east-1a.masters.saas.k8s.local
autoscaling-group	nodes.saas.k8s.local								nodes.saas.k8s.local
dhcp-options		saas.k8s.local									dopt-094ba5c9077359e63
iam-instance-profile	masters.saas.k8s.local								masters.saas.k8s.local
iam-instance-profile	nodes.saas.k8s.local								nodes.saas.k8s.local
iam-role		masters.saas.k8s.local								masters.saas.k8s.local
iam-role		nodes.saas.k8s.local								nodes.saas.k8s.local
instance		master-us-east-1a.masters.saas.k8s.local					i-0a40350cf54104a42
instance		nodes.saas.k8s.local								i-08a673a97be2e6d02
instance		nodes.saas.k8s.local								i-0bc315f50985fdda1
instance		nodes.saas.k8s.local								i-0eaed76a9fe7eb85b
internet-gateway	saas.k8s.local									igw-04753beb57c7c59af
keypair			kubernetes.saas.k8s.local-4e:96:75:47:1e:c8:9d:bf:07:e4:55:db:26:e8:b8:82	kubernetes.saas.k8s.local-4e:96:75:47:1e:c8:9d:bf:07:e4:55:db:26:e8:b8:82
load-balancer		api.saas.k8s.local								api-saas-k8s-local-c3t1l2
route-table		saas.k8s.local									rtb-0c9d25ec94e8660aa
security-group		api-elb.saas.k8s.local								sg-0293b44c7afba999f
security-group		masters.saas.k8s.local								sg-0906957f49af34f5a
security-group		nodes.saas.k8s.local								sg-0e6a8276a021d062a
subnet			us-east-1a.saas.k8s.local							subnet-0c5e7b1ff3e7db3b7
volume			a.etcd-events.saas.k8s.local							vol-04c0c3b209ac0162d
volume			a.etcd-main.saas.k8s.local							vol-07c8e4d3b002679b9
vpc			saas.k8s.local									vpc-0fb372784f88387fe

...

Deleted cluster: "saas.k8s.local"

```

and verify that the cluster has been terminated in your EC2. And remember, your cluster state is stored in the S3 bucket that you created! 

Delete S3 bucket via aws cli or AWS console.

```
$ aws s3 rm s3://k8s-saas-kops-state-dev/ --recursive
$ aws s3api delete-bucket --bucket k8s-saas-kops-state-dev --region us-east-1
```

And delete kops user and group using aws cli or AWS console.

Check:

https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:
https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LoadBalancers:
https://console.aws.amazon.com/iam/home?region=us-east-1#/home
https://console.aws.amazon.com/iam/home?region=us-east-1#/roles
https://s3.console.aws.amazon.com/s3/home?region=us-east-1
etc.



## mini-HOWTO: Using Terraform (Ref: https://github.com/kubernetes/kops/blob/master/docs/terraform.md)

```
------------------------kops edit cluster saas.k8s.local


# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
apiVersion: kops.k8s.io/v1alpha2
kind: Cluster
metadata:
  creationTimestamp: "2021-01-16T08:03:50Z"
  name: saas.k8s.local
spec:
  api:
    loadBalancer:
      type: Public
  authorization:
    rbac: {}
  channel: stable
  cloudProvider: aws
  configBase: s3://k8s-saas-kops-state-dev/saas.k8s.local
  containerRuntime: docker
  etcdClusters:
  - cpuRequest: 200m
    etcdMembers:
    - instanceGroup: master-us-east-1a
      name: a
    memoryRequest: 100Mi
    name: main
  - cpuRequest: 100m
    etcdMembers:
    - instanceGroup: master-us-east-1a
      name: a
    memoryRequest: 100Mi
    name: events
  iam:
    allowContainerRegistry: true
    legacy: false
  kubelet:
    anonymousAuth: false
  kubernetesApiAccess:
  - 0.0.0.0/0
  kubernetesVersion: 1.18.14
  masterInternalName: api.internal.saas.k8s.local
  masterPublicName: api.saas.k8s.local
  networkCIDR: 172.20.0.0/16
  networking:
    kubenet: {}
  nonMasqueradeCIDR: 100.64.0.0/10
  sshAccess:
  - 0.0.0.0/0
  subnets:
  - cidr: 172.20.32.0/19
    name: us-east-1a
    type: Public
    zone: us-east-1a
  topology:
    dns:
      type: Public
    masters: public
    nodes: public

------------------kops edit ig --name=saas.k8s.local nodes
apiVersion: kops.k8s.io/v1alpha2
kind: InstanceGroup
metadata:
  creationTimestamp: "2021-01-16T08:03:51Z"
  labels:
    kops.k8s.io/cluster: saas.k8s.local
  name: nodes
spec:
  image: 099720109477/ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-20201112.1
  machineType: t2.micro
  maxSize: 3
  minSize: 3
  nodeLabels:
    kops.k8s.io/instancegroup: nodes
  role: Node
  subnets:
  - us-east-1a

-------------kops edit ig --name=saas.k8s.local master-us-east-1a
apiVersion: kops.k8s.io/v1alpha2
kind: InstanceGroup
metadata:
  creationTimestamp: "2021-01-16T08:03:51Z"
  labels:
    kops.k8s.io/cluster: saas.k8s.local
  name: master-us-east-1a
spec:
  image: 099720109477/ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-20201112.1
  machineType: t2.micro
  maxSize: 1
  minSize: 1
  nodeLabels:
    kops.k8s.io/instancegroup: master-us-east-1a
  role: Master
  subnets:
  - us-east-1a
--------------------------
```

Set up remote state

You could keep your Terraform state locally, but we strongly recommend saving it on S3 with versioning turned on that bucket. 
Configure a remote S3 store with a setting like below:
```
terraform {
  backend "s3" {
    bucket = "mybucket"
    key    = "path/to/my/key"
    region = "us-east-1"
  }
}
```

earn more about Terraform state. 

Initialize/create a cluster

For example, a complete setup might be:
```
$ kops create cluster \
  --name=kubernetes.mydomain.com \
  --state=s3://mycompany.kubernetes \
  --dns-zone=kubernetes.mydomain.com \
  [... your other options ...]
  --out=. \
  --target=terraform
```
Example: 
```
kops create cluster --name="saas.k8s.local" --zones="us-east-1a" --master-size="t2.micro" --node-size="t2.micro" --node-count="3" --ssh-public-key="~/.ssh/k8s-saas.pub" --out=. --target=terraform
```
Initialize Terraform to set-up the S3 backend and provider plugins.
```
$ terraform init
$ terraform plan
$ terraform apply
```
Editing the cluster

It's possible to use Terraform to make changes to your infrastructure as defined by kOps. In the example below we'd like to change some cluster configs:
```
$ kops edit cluster \
  --name=kubernetes.mydomain.com \
  --state=s3://mycompany.kubernetes

# editor opens, make your changes ...
$ kops update cluster \
  --name=kubernetes.mydomain.com \
  --state=s3://mycompany.kubernetes \
  --out=. \
  --target=terraform
```
Then apply your changes after previewing what changes will be applied:
```
$ terraform plan
$ terraform apply
```
Teardown the cluster

When you eventually terraform destroy the cluster, you should still run kops delete cluster, to remove the kOps cluster specification and any dynamically created Kubernetes resources (ELBs or volumes). To do this, run:
```
$ terraform plan -destroy
$ terraform destroy
$ kops delete cluster --yes \
  --name=kubernetes.mydomain.com \
  --state=s3://mycompany.kubernetes
```
Ps: You don't have to kops delete cluster if you just want to recreate from scratch. Deleting kOps cluster state means that you've have to kops create again.

## mini-HOWTO: Using Terraform (Ref: https://github.com/kubernetes/kops/blob/master/docs/terraform.md)


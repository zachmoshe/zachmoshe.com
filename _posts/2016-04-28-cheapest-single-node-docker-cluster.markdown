---
layout: post
title: "Cheapest Single-Node Docker Cluster"
image: /images/articles/cheapest-docker/image.jpg
tags: ecs docker cluster
description: >
    How to launch the cheapest single-node Docker cluster? This post will compare fully managed docker clusters from Google (Google Container Engine) and Amazon (AWS Container Service). Since both products are similar in capabilities (or at least, cover my basic needs), the effort will be to try and get the cheapest node possible. That does the work for me mainly as a playground or for very tiny jobs/APIs.
---

If you have 20$-30$/month and you're willing to pay it for a server that you can run some stuff on, stop reading now and launch a [`t2.small`](https://aws.amazon.com/ec2/pricing/) or a [`g1-small`](https://cloud.google.com/compute/pricing#predefined_machine_types) machine. But if you're a cheap bastard like me, and want to go even lower than the 4.68$/month that a `t2.nano` machine would cost - keep reading...

First, just to clearly state the requirements, here is my story - I sometimes want to run either batch jobs or some simple APIs just for tests or my own use. Those are normally very small scripts that don't use much CPU or memory and if it's an API - it has almost no traffic at all but I still want it to be available 24/7.

I've used [AWS Labmda](https://aws.amazon.com/lambda/) for similar stuff, and while I'm basically a fan of this service, I find it annoying that I'm not free to implement however I want (they currently support Java, JS and Python only and I'm a big Ruby fan...). It's also not that easy to test and debug and it binds you to AWS.

[Docker](https://www.docker.com/) looks like a nice solution. I've played with it a couple of months ago and was quite impressed by how flexible I was in implementing my solution and how documented the outcome was. The `Dockerfile` file is probably the best installation instructions I could have ever ask for. Completely reproducible and not prone to 'oops-i-forgot-to-update' errors.

I'd also like to have a fully managed Docker service. Meaning - something that will support services (constantly running), tasks (scheduled or manually invoked) and handling cluster failures, reruns, etc... I can also live with a cluster that is not 100% available but I'd like it to be up most of the time (>95%) and more important - handle failures and shutdown automatically.

The most important thing - I'm not happy with paying a monthly fee for a server that will basically do nothing. However, I do understand that if I want a 24/7 API and don't want to use Lambda, I'll have to spin something up. So I'm willing to pay, but want the minimal monthly fee. And yes - every dollar counts.


## Compute Engine Candidates

### Google Container Engine

Google Cloud Platform is obviously a candidate when looking for computing power. Their [Container Engine](https://cloud.google.com/container-engine/) seems to offer something similar to what I'm after. A fully managed Docker cluster, built on [Kubernetes](http://kubernetes.io/) which comes with a bunch of extra features and runs on a well known compute engine.

The best thing seemed to be a single-node cluster of `g1-micro` instance (1CPU, 0.6GB memory). This should have cost 4.03$/month with [sustained use discount](https://cloud.google.com/compute/pricing#sustained_use). However, Google requires a minimum of 3 instances if choosing a `g1-micro` instance type, and that's too much for me.

The next available option is a `g1-small` instance (1CPU, 1.7GB memory) for 13.68$/month after sustained use discount.

Unfortunately, It seems like [preemptible instances](https://cloud.google.com/preemptible-vms/) are not supported with Container Service as they only last for up to 24 hours.


### AWS EC2 Container Service

Amazon supports long running preemptible instances with their [EC2 Spot Fleet](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-fleet.html) service. For me it's a huge improvement from the old [Spot Instances](https://aws.amazon.com/ec2/spot/) API because it keeps the request open forever rather than terminating after the first successful provisioning.

An `m3.medium` machine (1CPU, ~4GB memory) will cost 48.24$/month with the regular on-demand price, but could get as low as 9$/month for a spot instance bid that was above the line (wasn't interrupted) 100% of the last month.

Another interesting option for me (I basically need the cheapest machine) is the `t1.micro` machine (notice - `t1`, not `t2`). These machines give you 1CPU with 0.6GB memory (similar to Google's `g1-micro`) and with spot fleet I could get them running for a full month (based on last month's statistics) for **2.52$/month**. That's a winner!

The only problem is that t1 machines require a PV image and the 'regular' machines run on HVM image. Amazon's regular image for ECS comes as a HVM one.


## PV vs. HVM images

HVM stands for Hardware Virtual Machine and basically means that the hardware that runs your VM fully supports virtualization and therefore you can run your MBR (master boot record) without any changes as you would do on bare-metal hardware. If your hardware supports HVM and your OS knows that and uses the low-level APIs, you can get a performance boost, especially when it gets to network and GPU. HVM is currently the recommended image type to use on EC2 and supported by all the new instance types.

PV stands for paravirtual and can run even on hosts that don't explicitly support virtualization. However, it won't be able to use the special hardware accelerations. The old EC2 instance types support PV images.


## ECS compatible t1.micro image

So, `t1.micro` machines are available for cheap, but require a PV image to run. Amazon provides some ECS ready images, but those are all HVM images and don't support the t1 machine. Let's have a look on how ECS works and what is required from an instance in an ECS cluster and see if we can create ourselves a ECS-Compatible-PV-Image.

### ECS

ECS manages docker clusters. It support scheduling jobs and services to run on clusters and handles instances that join the cluster or suddenly terminate. Although it supports spinning machines from the console, that only uses the regular EC2 API and doesn't have anything to do with ECS itself. In fact, if we look at [ECS API](http://docs.aws.amazon.com/AmazonECS/latest/APIReference/Welcome.html) we can see that creating a cluster only create a cluster by its name and doesn't spin machines. Instances join a cluster when they run an [ECS agent](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_agent.html). The agent is the one to report to the cluster about the new instance (look at the [RegisterContainerInstance](http://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_RegisterContainerInstance.html) API call), get jobs to run and report metrics back.



### Build a PV image with ECS agent

First we'll create a role that allows ECS access:

* Create a new Role in the IAM console
* From 'AWS Service Roles' select 'Amazon EC2 Role for EC2 Container Service'

Now, let's launch an instance and install ECS agent on a regular PV image:

* Spin an instance with a PV AMI Linux AMI. The latest one (at the moment) is `Amazon Linux AMI 2016.03.0 x86_64 PV EBS` (`ami-a5368cd6`)
* Machine type doesn't really matter as we only use it to install some stuff and store the image. I used `m3.medium`
* Choose the Role you've created before
* Storage - I've used the regular 8GB for OS. You can use more but keep in mind that if your apps need more than just a temporary storage space, you shouldn't rely on the docker instance internal storage but use [Docker Volumes](https://docs.docker.com/engine/userguide/containers/dockervolumes/)
* Make sure you attach a security group that allows you to SSH to that machine
* Obviously - launch the machine with a keypair you have

After logging in, we'll follow the [ECS Agent Install tutorial](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-agent-install.html):

{% highlight console %}
sudo yum update -y
sudo yum install -y ecs-init
{% endhighlight %}

If you want to change any of the default configurations the ECS agent it's shipped with, create a file in `/etc/ecs/ecs.config` and set [the variables you want to override](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-agent-config.html).

For the cluster name, I recommend setting `ECS_CLUSTER` only at first boot (using user-data script) to allow multiple use of the same image.

After you're done - save the image ('Actions->Image->Create Image' from the Instances section in the console)



## Create a Cluster

We'll need to create a cluster before instances can join it. In the current version of the console, if it's your first run, you get a screen that does both cluster creation and spin instances together. Since we only want to create a cluster by name, we'll use the API instead:

{% highlight bash %}
aws ecs create-cluster --cluster-name test
# Notice that based on your AWS configuration, you might need to set region.
# For example: '--region eu-west-1'
{% endhighlight %}



## Launch Machines

* Open the [New Spot Console](https://eu-west-1.console.aws.amazon.com/ec2sp) and create a new spot request
* Choose 'Custom AMI' and use the one you've just saved
* Notice that with the PV image, `t1.micro` is available in the list
* Make sure you set appropriate **Security Groups** and **IAM Role** (In 'More Options')
* In user data, enter the following (obviously change `my_cluster_123`...):

{% highlight bash %}
#!/bin/bash -ex
echo 'ECS_CLUSTER=my_cluster_123' >> /etc/ecs/ecs.config
rm -fr /var/lib/ecs/data/*
{% endhighlight %}

* Review and Launch!

---
layout: page
title: AWS Tutorial
permalink: /aws-tutorial/For GPU instances, we also have an Amazon Machine Image (AMI) that you can use
---
对于 GPU(Graphics processing unit, 图形处理器。译者注：GPU包含于显卡。)实例，我们在Amazon EC2(Elastic Comput*e C*loud，亚马逊弹性计算云。译者注：和腾讯云，阿里云类似)有一台Amazon Machine Image (AMI, 亚马逊机器映像。译者注：类似于用于专门任务的云服务器或云虚拟主机)，我们可以使用它来启动 GPU 实例。本教程就是叫你如何在提供的 AMI 上建立你自己的 EC2 实例。**目前我们不提供给 CS231N 学生在 AWS 上的花费，但是很鼓励学生们在自己的预算经费内使用自己的 快照映像(snapshot) ** 

**概要 ** ：我们的映像是
`cs231n_caffe_torch7_keras_lasagne_v2`, AMI ID: `ami-125b2c72` 区域是 us-west-1
(美国西1)区。实例为 `g2.2xlarge` 。  Caffe, Torch7, Theano, Keras 和Lasagne
都已经预先安装。Caffe 的Python绑定也弄好了，有两个版本，一个CUDA 7.5 另一个是 CuDNN v3 可以使用。

首先，假如你还没有 AWS 账号，去 [AWS 官网](http://aws.amazon.com/), 点击 "Sign In to the
Console" 来注册，注册页面如下：

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/aws-signup.png'>
</div>

选择 "I am a new user/我是新用户" 复选框，点击 "Sign in using our secure
server/使用我们的安全服务器" 按钮注册，进行后面的操作。
随后会进行手机验证，和绑定你的信用卡信息。

注册成功之后，返回到[AWS](http://aws.amazon.com)主页，点击 "Sign In to the Console登录到控制台"，输入你的用户名和密码来登录:

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/aws-signin.png'>
</div>

进入之后，你会看到如下欢迎界面:

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/aws-homepage.png'>
</div>

确保右上角的区域信息为北加州/N. California，如果不是点击从下拉列表中选择该项。 (注意，以下操作的前提是你的账户已经被亚马逊验证过了，通常验证时长位两小时左右，在被验证完成之前无法启用你的实例)

然后，点击 EC2 链接 (归属于 Compute/计算类)。你将会看到如下图所示的仪表盘:

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/ec2-dashboard.png'>
</div>

点击 "Launch Instance/启动实例"，就会被重定向到如下图所示的页面：

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/ami-selection.png'>
</div>

点击左侧边栏的 "Community/公共 AMIs" 链接，在搜索框中输入 "cs231n"
进行查找，你就能找到我们的 AMI
`cs231n_caffe_torch7_keras_lasagne_v2` (AMI ID: `ami-125b2c72`)。选择该
AMI，下一步是选择你的实例类型：

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/community-AMIs.png'>
</div>

选择实例类型 `g2.2xlarge`, 接着点击 "Review and Launch/检查和启动"。

<div class='fig figcenter fighighlight'>  <img src='/assets/aws-tutorial/instance-selection.png'>

</div>

跳转到下一个界面，点击启动。

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/launch-screen.png'>
</div>

提示创建或者使用一个已存在的密钥对(key-pair)，如果你使用过并且有了密钥对就使用原来的，否则选择从下拉菜单中选择 "Create a new key pair/创建一个新的密钥对"，接着你就可以下载这个密钥对，然后把它存储在电脑上一个你暂时不会删除的地方，假如你的密钥丢失，你就**永远没有办法再访问你的实例** :

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/key-pair.png'>
</div>

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/key-pair-create.png'>
</div>

下载好你的密钥后，你应当改变他的读写权限为“user-only/仅用户个人”，在 Linux/OSX 下你可以输入命令如下：

```
$ chmod 600 PEM_FILENAME
```
这里`PEM_FILENAME` 代表你刚才下载的 .pem 的文件全称。

完成后，点击"Launch Instances/启动实例"，就可以看到一个实例正在启动的画面：

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/launching-screen.png'>
</div>

点击 "View Instances/查看实例" 来查看你实例的状态，其状态有
"Running/正在运行" 和 "2/2 status checks passed/状态检查通过" 。现在你可以使用 ssh(Secure Shell，是一种安全协议)来访问你的实例：

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-tutorial/instances-page.png'>
</div>

首先，从实例列表中记下你的 Public IP(公共网络协议地址)，然后之执行：

```
ssh -i PEM_FILENAME ubuntu@PUBLIC_IP
```

现在你可以登录到你的实例中，输入以下命令检查Caffe是否能够正常工作：

```
$ cd caffe
$ ./build/tools/caffe time --gpu 0 --model examples/mnist/lenet.prototxt
```

Caffe, Theano, Torch7, Keras 和Lasagne 都已经余弦安装好了。默认 Caffe 和 python进行了绑定，并安装了两个版本CUDA 7.5 和CuDNN v3。如果有类似以下的错误发生：

```
Check failed: error == cudaSuccess (77 vs.  0)  an illegal memory access was encountered
```

你可能会终止和重启实例，这种错误非常少见，我(助教)也不明白问题出在哪里。

关于如何使用实例，你要知道：

- 根文件的容量有12GB， 但只有大约 3GB 是可以使用的。
- 你至多可以把 60GB 的数据、模型、检查点放到 `/mnt` 路径中。
- 重启和终止，有时 `/mnt` 文件夹并非每次都有。
- 在不使用实例的时候将其关闭，否则会持续计费。运行GPU实例是比较昂贵的，所以请合理安排你的经费。 (磁盘存储也是需要计费的，特别是你购买了比较大的磁盘空间的时候)。
- 可以学习自定义提示，当你的实例什么都没做的时候自动停止运行。
- 如果每次启用你的实例来访问一个大的数据它，而你不想每次都下载它，最好的方式是创建一个AMI, 并配置AMI与你的机器相关联(这个环节是在你选好AMI后，启用它之前)。

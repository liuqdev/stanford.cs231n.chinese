---
layout: page
title: IPython Tutorial
permalink: /ipython-tutorial/
---

***(注意: 文档中的部分截图可能有些过时了，然而这不妨碍将本文视作是为对一般菜单布局和某些功能等等的快速介绍。)***

在本门课中，我们将是用到 IPython notebook (最近换了身马甲叫做
[Jupyter notebook](https://jupyter.org/)) 来提交编程任务。
有了 IPython notebook，我们就可以在浏览器中编写和执行Python 代码。
你能够在 IPython notebooks 中前一块后一片(in bits 
and pieces) 地修订和执行你的代码。正是由于这个原因，IPython notebook 广泛使用于科学计算中：

*(注意: 如果你的虚拟环境安装正确(因为每个编程任务提交都需要)，那么你就不需要按照网站上的指示来安装。只需要到你编程任务的文件夹下运行`source .env/bin/activate` 即可。*

一旦你[安装](http://jupyter.org/install.html)了 notebook，输入下方命令来启动:

```
jupyter notebook
```

一旦你的notebook 服务器成功运行，你的浏览器就指向了 http://localhost:8888 就可以开始使用 notebook 了。一切正确的话，你会看到如下方所示的画面，显示当前路径下所有可用的 IPython notebook：

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/file-browser.png'>
</div>

如果点击一个 notebook 文件，你会看到如下的画面：

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-1.png'>
</div>

一个 IPython notebook 由若干的**胞/小单元(cell)** 组成。每个胞/小单元(cell)中可以是一个Python 代码块，点击运行或者按下组合键`Shift+Enter`都可以(独立)运行该代码块，该代码块运行后的输出会显示在此胞/小单元(cell)的下方。举例如下：

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-2.png'>
</div>

全局变量为所有胞/小单元(cell)所共有。因此下面的例子运行会得到如下的结果：

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-3.png'>
</div>

根据惯例，IPython notebook会从上至下执行。执行某些胞/小单元(cell)时出错，或者没有按照正确的顺序执行，会产生一下错误：

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-error.png'>
</div>

当你更改、执行了你要提交编程任务的notebook 中的某些胞/小单元(cell)，记住**要保存你的更改** 。

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/save-notebook.png'>
</div>

这里只是对 IPython notebook 作了简要的介绍, 但是对于本门课程编程练习而言以及很够用了。

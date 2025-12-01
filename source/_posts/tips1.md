---
title: Ubuntu在Anaconda环境中安装包时报错：OSError：[Errno 28] 设备上没有空间
author: luvisdru9
date: 2025-05-29 00:00:00
updated: 2025-05-29 00:00:00
tags: 
  - Ubuntu
  - Anaconda
categories: Solution
description: Ubuntu在Anaconda环境中安装包时报错：OSError：[Errno 28] 设备上没有空间
keywords:
  - Ubuntu
  - Anaconda
  - Error
#top_img:
#comments:
#cover:
#toc:
#toc_number:
#toc_style_simple:
#copyright:
#copyright_author:
#copyright_author_href:
#copyright_url:
#copyright_info:
#mathjax:
#katex:
#aplayer:
#highlight_shrink:
#aside:
#abcjs:
#noticeOutdate:
---

今天在本地部署SAM2，conda创建环境后安装torch相关依赖包，但是安装到一半报错如下：

```bash
ERROR: Exception:
Traceback (most recent call last):
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
    yield
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py", line 561, in read
    data = self._fp_read(amt) if not fp_closed else b""
           ^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
           ^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/cachecontrol/filewrapper.py", line 102, in read
    self.__buf.write(data)
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/tempfile.py", line 499, in func_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 28] 设备上没有空间

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/cli/base_command.py", line 105, in _run_wrapper
    status = _inner_run()
             ^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/cli/base_command.py", line 96, in _inner_run
    return self.run(options, args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/cli/req_command.py", line 68, in wrapper
    return func(self, options, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/commands/install.py", line 387, in run
    requirement_set = resolver.resolve(
                      ^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 182, in resolve
    self.factory.preparer.prepare_linked_requirements_more(reqs)
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 559, in prepare_linked_requirements_more
    self._complete_partial_requirements(
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 474, in _complete_partial_requirements
    for link, (filepath, _) in batch_download:
                               ^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/network/download.py", line 313, in __call__
    filepath, content_type = self._downloader(link, location)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/network/download.py", line 185, in __call__
    bytes_received = self._process_response(
                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/network/download.py", line 208, in _process_response
    return self._write_chunks_to_file(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/network/download.py", line 218, in _write_chunks_to_file
    for chunk in chunks:
                 ^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/cli/progress_bars.py", line 61, in _rich_download_progress_bar
    for chunk in iterable:
                 ^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_internal/network/utils.py", line 65, in response_chunks
    for chunk in response.raw.stream(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py", line 622, in stream
    data = self.read(amt=amt, decode_content=decode_content)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py", line 560, in read
    with self._error_catcher():
         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/home/lzm/anaconda3/envs/SAM2/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py", line 455, in _error_catcher
    raise ProtocolError("Connection broken: %r" % e, e)
pip._vendor.urllib3.exceptions.ProtocolError: ("Connection broken: OSError(28, '设备上没有空间')", OSError(28, '设备上没有空间'))
```

使用``df -h``查看，发现根分区已占用 95%（仅剩 2.7G），而Python包安装需要临时空间（在/tmp），因此操作失败。因为没有剩下的未分配空间，扩展操作起来比较麻烦，所以使用了以下临时解决方法，实测有效：
- 在内存充足的``home``目录新建一个tmp文件夹
- 打开一个终端，使用命令``export TMPDIR=$HOME/tmp``（不能换终端，该命令为临时的）
- 而后就可以开始下载，临时文件会改为存储在``/home/tmp``中
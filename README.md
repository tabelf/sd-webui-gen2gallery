# ai-gallery
<hr/>

[ English | [Chinese](README_zh.md) ]

ai-gallery is a front-end and back-end separation system based on Go-Zero + SD Plugin + Vite/React + Ant Design technology, which is used to uniformly manage SD painting tasks.

- [Backend Project](https://github.com/tabelf/ai-gallery) <br/>
- [Front-end Project](https://github.com/tabelf/ai-gallery-ui) <br/>
- [SD Plugin](https://github.com/tabelf/sd-webui-gen2gallery) <br/>

## Demo
<hr/>

https://github.com/user-attachments/assets/db6ac661-84ca-47b8-934b-86f1f61a9578


[Full video demonstration](https://www.bilibili.com/video/BV1mt8ue6E1Y)

## ✨ Feature
<hr/>

- Manage SD txt2img and img2img drawing task tecord
- SD generate image cloud/local storage
- Aggregation of task data submitted by multiple users
- User management
- System settings

## 💼 Preparation

### Environment Preparation
```text
go sdk                   v1.21.0
node                     v21.6.2
npm                      v9.6.7
react                    v18.2.0
stable diffuison webui   v1.9.3
mysql                    v8.0.28
nginx                    v1.25.3
```

### Get the code
```bash
# get the plugin project code 
git clone https://github.com/tabelf/sd-webui-gen2gallery.git

# get the backend project code
git clone https://github.com/tabelf/ai-gallery.git

# get front-end project code
git clone https://github.com/tabelf/ai-gallery-ui.git
```

## Startup

### Server startup
```bash
# enter the backend project directory
cd ./ai-gallery

# install go dependencies
go mod tidy

# modify database config
vi ./service/etc/application-dev.yaml

# root: database username
# 12345678: database password
# ai_gallery: database name

# db:
#  url: root:12345678@(127.0.0.1:3306)/ai_gallery?charset=utf8mb4&parseTime=true&loc=Local&interpolateParams=true
#  driver: mysql
#  max_open_conns: 100
#  max_idle_conns: 30

# generate database tables
go run cmd/main.go migrate --env dev

# run
go run cmd/main.go start --env dev
```

### Front-end startup
```bash
# install dependencies. 
npm install

# start the service
vite dev

# the address will be displayed if the deployment is successful
http://localhost:5173/

# login
# username：admin 
# password：   1234567

# Go to the system settings menu bar and configure the location where the pictures are to be stored
# If you choose local storage, you need to configure the following nginx
# If you choose Tencent Cloud cos for storage, you need to configure the storage address, ID, key, and remember to turn on cross-domain settings
# If you choose Alibaba Cloud OOS for storage, you need to configure the bucket name, storage address, ID, key, and remember to enable cross-domain settings.
```

### Login Account
```
username：admin 
password：1234567
```

### SD Plugin
```
# 1. Put the plug-in project in the extensions directory of stable diffusion webui

# 2. Normal start ./webui
```

### Nginx config
```
# Add in the nginx.conf configuration file

location /upload/ {
    alias /Users/stable-diffusion-webui/; # 修改成自己 stable-diffusion-webui 文件的路径亘路径
    autoindex on;    

    add_header 'Access-Control-Allow-Origin' '*';
    add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, HEAD, OPTIONS';
    add_header 'Access-Control-Allow-Headers' 'Content-Type, X-CSRF-Token, Authorization, AccessToken, Token, Cache-Control';
    add_header 'Access-Control-Allow-Credentials' 'true';

    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, HEAD, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Content-Type, X-CSRF-Token, Authorization, AccessToken, Token, Cache-Control';
        add_header 'Access-Control-Allow-Credentials' 'true';
        add_header 'Access-Control-Max-Age' 1728000;
        add_header 'Content-Type' 'text/plain charset=UTF-8';
        add_header 'Content-Length' 0;
        return 204;
     }
}
```

## 联系

<table>
   <tr>
    <td><img src="./wechat.png" width="180px"></td>
  </tr>
  <tr>
    <td>wechat</td>
  </tr>
</table>

<template>
  <div class="posts">
      <div class="load">
          <router-link class="link" :to="{name: 'home'}">See more</router-link>
      </div>
  </div>
</template>

<script>
export default {
    name: 'posts'
}
</script>

<style lang="scss" scoped>
.posts {
    width: 100%;
    max-width: 1280px;
    margin: 50px auto;

    .load {
        display: block;
        width: 100%;
        max-width: 250px;
        height: 50px;
        margin: 0 auto;
        outline: none;
        border: none;
        border-radius: 15px;
        appearance: none;

        background-color: #FFCE00;
        box-shadow: 3px 3px 8px 0px rgba(0,0,0,0.2);

        color: #171717;
        font-size: 28px;
        padding-top: 7px;
        box-sizing: border-box;
        text-align: center;
        cursor: pointer;
        
        a {
            text-transform: none;
            text-decoration: none;
            color: #171717;
            font-size: 28px;
            padding-top: 7px;
            box-sizing: border-box;
            text-align: center;
            cursor: pointer; 
        }
    }
}
</style>
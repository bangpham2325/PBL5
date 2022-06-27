<template>
  <div class="intro">
      <h3>System Functions</h3>
      <div class="posts">
          <div class="post row-2 col-2">
              <router-link to='/home'>
                    <div class="image" :style="{backgroundImage: `url('${require('../assets/traffic1.jpg')}')`}"></div>
                    <h4>Vehicle Identification</h4> 
              </router-link>
          </div>
          <div class="post row-2" style="grid-column: 3/6">
              <router-link to='/classification'>
                    <div class="image" :style="{backgroundImage: `url('${require('../assets/traffic2.jpg')}')`}"></div>
                    <h4>Vehicle Speed Prediction</h4>
              </router-link>
          </div>
          <div class="post post-especial"  style="grid-row: 3/6">
              <router-link to='/statistics'>
                    <div class="image" :style="{backgroundImage: `url('${require('../assets/traffic8.png')}')`}"></div>
                    <h4>Vehicle Analysics</h4>
              </router-link>
          </div>
          <div class="post row-2" style="grid-column: 1/4">
              <router-link to='/violation'>
                    <div class="image" :style="{backgroundImage: `url('${require('../assets/traffic6.jpg')}')`}"></div>
                    <h4>Vehicle Violation</h4>
              </router-link>
          </div>
          <div class="post row-2 col-2">
              <router-link to='/home'>
                    <div class="image" :style="{backgroundImage: `url('${require('../assets/traffic9.jpg')}')`}"></div>
                    <h4>Future Development</h4>
              </router-link>
          </div>
      </div>
  </div>
</template>

<script>
export default {
    name: 'Intro',


}
</script>

<style lang="scss" scoped>
.intro {
    display: block;
    width: 100%;
    max-width: 1280px;
    margin: 0 auto;
    .posts {
        display: grid;
        grid-template-columns: repeat(5,1fr);
        grid-template-rows: repeat(7,196px);
        grid-gap: 30px;
        margin: 15px;
        .post {
            display:flex;
            flex-direction: column;
            background-color: #FFFFFF;
            box-shadow: 3px 3px 8px 0px rgba(0,0,0,0.2);
            cursor: pointer;
            padding: 12px;
            a {
                width: 100%;
                height: 100%;
                color: inherit;
                text-decoration: none;
                .image {
                    width: 100%;
                    height: 90%;
                    background-position: center;
                    background-size: cover;
                    background-repeat: no-repeat;
                    flex: 1;
                }
                h4 {
                    box-sizing: border-box;
                    padding: 3px 15px 0px 15px;
                    height: 50px;
                    line-height: 50px;
                    margin: 0px;
                }
            }

            &.col-2 {
                grid-column: span 2;
            }
            &.row-2 {
                grid-row: span 2;
            }
        }
        .col-2 {
            max-width: none!important;
        }
        .col-6 {
            max-width: none!important;
        }
        .post-especial {
            grid-column-start: 1;
            grid-column-end: 6;
        }
    }
}
</style>
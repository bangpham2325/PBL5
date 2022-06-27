<template>
 <div id="app">
    <component :is="layout">
      <router-view />
    </component>
  </div>
</template>

<script>
const defaultLayout = 'default'
export default {
  name: 'App',

  data: () => ({
    //
  }),
  computed: {
    layout() {
      return (this.$route.meta.layout||defaultLayout) + '-layout'
    }
  }
};
</script>
<style lang="scss" scoped>
* {
  padding: 0px;
  margin: 0px;
  width: 100vw;
  height: 100vh;
}
</style>
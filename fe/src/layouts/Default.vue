<template>
  <div class="default">
    <Navbar />
    <main class="main">
        <slot class="slot-main"/>
    </main>
  </div>
</template>

<script>
import Navbar from '@/components/Navbar.vue'
export default {
  components: { Navbar },

}
</script>

<style lang="scss" scoped>
main {
  padding-top: 95px;
}
</style>
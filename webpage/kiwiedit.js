// ======================== Gallery Tab Switching =====================
document.addEventListener('DOMContentLoaded', function() {
    initGalleryTabs();
    initVideoComparisons();
    initVideoCarousels();
});

var videoPlaybackSystemInitialized = false;
var videoLoadObserver = null;
var videoVisibilityObserver = null;

function initGalleryTabs() {
    // Initialize all tab groups on the page
    var tabGroups = document.querySelectorAll('.gallery-tabs');
    tabGroups.forEach(function(tabGroup) {
        var groupId = tabGroup.getAttribute('data-gallery');
        var tabs = tabGroup.querySelectorAll('.gallery-tab');
        var contents = document.querySelectorAll('.gallery-content[data-gallery="' + groupId + '"]');

        tabs.forEach(function(tab) {
            tab.addEventListener('click', function() {
                var target = this.getAttribute('data-target');

                // Deactivate all tabs in this group
                tabs.forEach(function(t) { t.classList.remove('active'); });
                // Activate clicked tab
                this.classList.add('active');

                // Hide all contents in this group
                contents.forEach(function(c) { c.classList.remove('active'); });
                // Show target content
                var targetContent = document.querySelector('.gallery-content[data-gallery="' + groupId + '"][data-id="' + target + '"]');
                if (targetContent) {
                    targetContent.classList.add('active');
                    preloadActiveGalleryVideos(targetContent);
                }
                refreshManagedVideoPlayback();
            });
        });
    });
}

function initVideoComparisons() {
    if (!window.customElements) {
        initVideoPlaybackSystem();
        return;
    }

    var convertPairs = function() {
        try {
            var pairs = document.querySelectorAll('.video-pair');

            pairs.forEach(function(pair) {
                var columns = pair.children;
                if (!columns || columns.length < 2) {
                    return;
                }

                var firstBlock = columns[0];
                var secondBlock = columns[1];
                var firstMedia = firstBlock.querySelector('video, img');
                var secondMedia = secondBlock.querySelector('video, img');

                if (!firstMedia || !secondMedia) {
                    return;
                }

                var firstLabelNode = firstBlock.querySelector('.pair-label');
                var secondLabelNode = secondBlock.querySelector('.pair-label');
                var firstLabel = firstLabelNode ? firstLabelNode.textContent.trim() : 'Input';
                var secondLabel = secondLabelNode ? secondLabelNode.textContent.trim() : 'Output';
                var wrapper = buildComparisonSlider(firstMedia, secondMedia, firstLabel, secondLabel);
                if (!wrapper) {
                    return;
                }
                pair.replaceWith(wrapper);
            });

            var referenceTriples = document.querySelectorAll('.gallery-content[data-gallery="reference"] .video-triple');
            referenceTriples.forEach(function(triple) {
                var blocks = triple.children;
                if (!blocks || blocks.length < 3) {
                    return;
                }

                var refBlock = blocks[0];
                var inputBlock = blocks[1];
                var outputBlock = blocks[2];
                var refMedia = refBlock.querySelector('video, img');
                var inputMedia = inputBlock.querySelector('video, img');
                var outputMedia = outputBlock.querySelector('video, img');
                if (!refMedia || !inputMedia || !outputMedia) {
                    return;
                }

                var refLabelNode = refBlock.querySelector('.pair-label');
                var inputLabelNode = inputBlock.querySelector('.pair-label');
                var outputLabelNode = outputBlock.querySelector('.pair-label');
                var refLabel = refLabelNode ? refLabelNode.textContent.trim() : 'Reference';
                var inputLabel = inputLabelNode ? inputLabelNode.textContent.trim() : 'Input';
                var outputLabel = outputLabelNode ? outputLabelNode.textContent.trim() : 'Output';
                var compareWrapper = buildComparisonSlider(inputMedia, outputMedia, inputLabel, outputLabel);
                var refWrapper = buildReferenceMedia(refMedia, refLabel);
                if (!compareWrapper || !refWrapper) {
                    return;
                }

                var layout = document.createElement('div');
                layout.className = 'reference-compare';

                var refPanel = document.createElement('div');
                refPanel.className = 'reference-compare-ref';
                refPanel.appendChild(refWrapper);

                var sliderPanel = document.createElement('div');
                sliderPanel.className = 'reference-compare-slider';
                sliderPanel.appendChild(compareWrapper);

                layout.appendChild(refPanel);
                layout.appendChild(sliderPanel);
                triple.replaceWith(layout);
            });

            var grids = document.querySelectorAll('.video-grid');
            grids.forEach(function(grid) {
                updateEdgeSpacers(grid);
                var index = parseInt(grid.dataset.carouselIndex || '0', 10);
                if (Number.isNaN(index)) {
                    index = 0;
                }
                scrollToVideoIndex(grid, index, false);
            });
        } finally {
            initVideoPlaybackSystem();
        }
    };

    if (window.customElements.get('img-comparison-slider')) {
        convertPairs();
        return;
    }

    window.customElements.whenDefined('img-comparison-slider').then(function() {
        convertPairs();
    });
    setTimeout(function() {
        if (!window.customElements.get('img-comparison-slider')) {
            initVideoPlaybackSystem();
        }
    }, 2200);
}

function buildReferenceMedia(media, label) {
    if (!media) {
        return null;
    }

    prepareMedia(media);

    var wrapper = document.createElement('div');
    wrapper.className = 'video-comparison-wrap reference-media-wrap';
    ensureLoadingOverlay(wrapper);
    wrapper.appendChild(media);

    var legend = document.createElement('span');
    legend.className = 'comparison-legend left';
    legend.textContent = label || 'Reference';
    wrapper.appendChild(legend);
    return wrapper;
}

function buildComparisonSlider(firstMedia, secondMedia, firstLabel, secondLabel) {
    if (!firstMedia || !secondMedia) {
        return null;
    }

    prepareMedia(firstMedia);
    prepareMedia(secondMedia);

    var slider = document.createElement('img-comparison-slider');
    slider.className = 'slider-container video-comparison';

    var before = document.createElement('figure');
    before.setAttribute('slot', 'first');
    before.className = 'before';
    before.appendChild(firstMedia);

    var after = document.createElement('figure');
    after.setAttribute('slot', 'second');
    after.className = 'after';
    after.appendChild(secondMedia);

    var wrapper = document.createElement('div');
    wrapper.className = 'video-comparison-wrap';
    ensureLoadingOverlay(wrapper);

    var leftLegend = document.createElement('span');
    leftLegend.className = 'comparison-legend left';
    leftLegend.textContent = firstLabel || 'Input';

    var rightLegend = document.createElement('span');
    rightLegend.className = 'comparison-legend right';
    rightLegend.textContent = secondLabel || 'Output';

    slider.appendChild(before);
    slider.appendChild(after);
    wrapper.appendChild(slider);
    wrapper.appendChild(leftLegend);
    wrapper.appendChild(rightLegend);

    syncComparisonPair(firstMedia, secondMedia);
    return wrapper;
}

function prepareMedia(media) {
    if (media.tagName !== 'VIDEO') {
        return;
    }

    if (media.dataset.prepared === 'true') {
        return;
    }

    media.dataset.prepared = 'true';
    media.setAttribute('playsinline', '');
    media.setAttribute('preload', 'none');
    media.removeAttribute('autoplay');
    media.muted = true;
    media.autoplay = false;
    media.loop = true;
    media.playsInline = true;
    media.controls = false;

    var src = media.getAttribute('src');
    if (src && !media.dataset.src) {
        media.dataset.src = src;
    }
    if (media.getAttribute('src')) {
        media.removeAttribute('src');
        media.load();
    }
}

function syncComparisonPair(firstMedia, secondMedia) {
    if (firstMedia.tagName !== 'VIDEO' || secondMedia.tagName !== 'VIDEO') {
        return;
    }

    var leader = firstMedia;
    var follower = secondMedia;
    var isBuffering = false;
    var isStarting = false;
    var wrapper = leader.closest('.video-comparison-wrap');

    leader.dataset.syncRole = 'leader';
    follower.dataset.syncRole = 'follower';

    function bothReadyToStart() {
        return leader.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
            follower.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
    }

    function playVideoSafely(video) {
        var p = video.play();
        if (p && typeof p.catch === 'function') {
            return p.catch(function() {});
        }
        return Promise.resolve();
    }

    function pauseBoth() {
        if (!leader.paused) {
            leader.pause();
        }
        if (!follower.paused) {
            follower.pause();
        }
    }

    function markBuffering() {
        if (isBuffering) {
            return;
        }
        isBuffering = true;
        pauseBoth();
    }

    function startSynchronizedPlayback(forceSeek) {
        if (!canVideoAutoPlay(leader) || isStarting || !bothReadyToStart()) {
            pauseBoth();
            return;
        }
        isStarting = true;

        var targetTime = leader.currentTime;
        if (forceSeek) {
            hardAlign(targetTime);
        }
        follower.playbackRate = leader.playbackRate;

        Promise.all([playVideoSafely(leader), playVideoSafely(follower)]).finally(function() {
            isStarting = false;
            isBuffering = false;
            updateLoadingState(wrapper);
        });
    }

    function hardAlign(targetTime) {
        var clamped = targetTime;
        if (clamped < 0) {
            clamped = 0;
        } else if (Number.isFinite(follower.duration)) {
            clamped = Math.min(clamped, Math.max(0, follower.duration - 0.001));
        }
        try {
            follower.currentTime = clamped;
        } catch (e) {
            // Ignore occasional seek errors before metadata is ready.
        }
    }

    function alignNow(forceSeek) {
        if (isStarting) {
            return;
        }

        if (!canVideoAutoPlay(leader)) {
            pauseBoth();
            return;
        }

        if (isBuffering) {
            startSynchronizedPlayback(true);
            return;
        }

        if (!bothReadyToStart()) {
            markBuffering();
            updateLoadingState(wrapper);
            return;
        }

        if (leader.paused) {
            startSynchronizedPlayback(forceSeek);
            return;
        }

        if (follower.paused) {
            startSynchronizedPlayback(true);
            return;
        }

        follower.playbackRate = leader.playbackRate;
        var targetTime = leader.currentTime;
        var drift = targetTime - follower.currentTime;
        var absDrift = Math.abs(drift);

        // Direct time lock (no keyframe-style correction).
        if (forceSeek || absDrift > 0.08) {
            hardAlign(targetTime);
        }
        updateLoadingState(wrapper);
    }

    ['loadedmetadata', 'play', 'pause', 'seeking', 'seeked', 'ratechange', 'ended'].forEach(function(evt) {
        leader.addEventListener(evt, function() {
            alignNow(true);
        });
    });

    ['waiting', 'stalled', 'suspend'].forEach(function(evt) {
        leader.addEventListener(evt, markBuffering);
        follower.addEventListener(evt, markBuffering);
    });

    ['canplay', 'canplaythrough', 'playing'].forEach(function(evt) {
        leader.addEventListener(evt, function() {
            if (isBuffering) {
                startSynchronizedPlayback(true);
            }
            updateLoadingState(wrapper);
        });
        follower.addEventListener(evt, function() {
            if (isBuffering) {
                startSynchronizedPlayback(true);
            }
            updateLoadingState(wrapper);
        });
    });

    leader.addEventListener('loadeddata', function() { updateLoadingState(wrapper); });
    follower.addEventListener('loadeddata', function() { updateLoadingState(wrapper); });
    leader.addEventListener('error', function() { updateLoadingState(wrapper); });
    follower.addEventListener('error', function() { updateLoadingState(wrapper); });

    var timer = setInterval(function() {
        if (!leader.isConnected || !follower.isConnected) {
            clearInterval(timer);
            return;
        }
        alignNow(false);
    }, 120);

    if (bothReadyToStart()) {
        startSynchronizedPlayback(true);
    } else {
        markBuffering();
        updateLoadingState(wrapper);
    }

    leader._syncAlign = alignNow;
}

function initVideoCarousels() {
    var grids = document.querySelectorAll('.video-grid');

    grids.forEach(function(grid) {
        if (grid.dataset.carouselInit === 'true') {
            return;
        }

        grid.dataset.carouselInit = 'true';
        var itemCount = grid.querySelectorAll('.video-item').length;

        var row = document.createElement('div');
        row.className = 'video-row';
        grid.parentNode.insertBefore(row, grid);
        row.appendChild(grid);

        var prevBtn = document.createElement('button');
        prevBtn.className = 'video-nav prev';
        prevBtn.type = 'button';
        prevBtn.setAttribute('aria-label', 'Previous videos');
        prevBtn.innerHTML = '&#10094;';

        var nextBtn = document.createElement('button');
        nextBtn.className = 'video-nav next';
        nextBtn.type = 'button';
        nextBtn.setAttribute('aria-label', 'Next videos');
        nextBtn.innerHTML = '&#10095;';

        prevBtn.addEventListener('click', function() {
            scrollVideoGrid(grid, -1);
        });

        nextBtn.addEventListener('click', function() {
            scrollVideoGrid(grid, 1);
        });

        if (itemCount > 1) {
            row.appendChild(prevBtn);
            row.appendChild(nextBtn);
        }

        grid.dataset.carouselIndex = '0';
        attachCarouselSync(grid);
        updateEdgeSpacers(grid);
        scrollToVideoIndex(grid, 0, false);
        setTimeout(function() {
            updateEdgeSpacers(grid);
            scrollToVideoIndex(grid, 0, false);
        }, 80);
    });
}

function scrollVideoGrid(grid, direction) {
    var items = grid.querySelectorAll('.video-item');
    var total = items.length;
    if (!total) {
        return;
    }

    syncCarouselIndex(grid);
    var index = parseInt(grid.dataset.carouselIndex || '0', 10);
    if (Number.isNaN(index)) {
        index = 0;
    }

    index = (index + direction + total) % total;
    grid.dataset.carouselIndex = String(index);
    scrollToVideoIndex(grid, index, true);
}

function scrollToVideoIndex(grid, index, smooth) {
    var items = grid.querySelectorAll('.video-item');
    if (!items.length || index < 0 || index >= items.length) {
        return;
    }

    var item = items[index];
    var targetLeft = item.offsetLeft - (grid.clientWidth - item.clientWidth) / 2;
    var maxScroll = Math.max(0, grid.scrollWidth - grid.clientWidth);
    var clampedLeft = Math.min(maxScroll, Math.max(0, targetLeft));
    grid.scrollTo({
        left: clampedLeft,
        behavior: smooth ? 'smooth' : 'auto'
    });
}

function attachCarouselSync(grid) {
    var timer = null;

    grid.addEventListener('scroll', function() {
        if (timer) {
            clearTimeout(timer);
        }
        timer = setTimeout(function() {
            syncCarouselIndex(grid);
        }, 120);
    });

    window.addEventListener('resize', function() {
        updateEdgeSpacers(grid);
        var index = parseInt(grid.dataset.carouselIndex || '0', 10);
        if (Number.isNaN(index)) {
            index = 0;
        }
        scrollToVideoIndex(grid, index, false);
    });
}

function syncCarouselIndex(grid) {
    var items = grid.querySelectorAll('.video-item');
    if (!items.length) {
        return;
    }

    var viewportCenter = grid.scrollLeft + grid.clientWidth / 2;
    var nearestIndex = 0;
    var nearestDistance = Infinity;

    items.forEach(function(item, idx) {
        var center = item.offsetLeft + item.clientWidth / 2;
        var distance = Math.abs(center - viewportCenter);
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestIndex = idx;
        }
    });

    grid.dataset.carouselIndex = String(nearestIndex);
}

function updateEdgeSpacers(grid) {
    var items = grid.querySelectorAll('.video-item');
    if (!items.length) {
        return;
    }

    var startSpacer = getOrCreateEdgeSpacer(grid, 'start');
    var endSpacer = getOrCreateEdgeSpacer(grid, 'end');

    var styles = window.getComputedStyle(grid);
    var gap = parseFloat(styles.gap || styles.columnGap || '0');
    if (Number.isNaN(gap)) {
        gap = 0;
    }

    var firstWidth = items[0].getBoundingClientRect().width;
    var lastWidth = items[items.length - 1].getBoundingClientRect().width;
    var startWidth = Math.max(0, (grid.clientWidth - firstWidth) / 2 - gap);
    var endWidth = Math.max(0, (grid.clientWidth - lastWidth) / 2 - gap);

    startSpacer.style.flex = '0 0 ' + startWidth + 'px';
    endSpacer.style.flex = '0 0 ' + endWidth + 'px';
}

function getOrCreateEdgeSpacer(grid, side) {
    var spacer = grid.querySelector('.video-edge-spacer.' + side);
    if (spacer) {
        return spacer;
    }

    spacer = document.createElement('div');
    spacer.className = 'video-edge-spacer ' + side;
    spacer.setAttribute('aria-hidden', 'true');
    if (side === 'start') {
        grid.insertBefore(spacer, grid.firstChild);
    } else {
        grid.appendChild(spacer);
    }
    return spacer;
}

function initVideoPlaybackSystem() {
    if (videoPlaybackSystemInitialized) {
        refreshManagedVideoPlayback();
        return;
    }
    videoPlaybackSystemInitialized = true;

    if (!('IntersectionObserver' in window)) {
        var all = document.querySelectorAll('video');
        all.forEach(function(video) {
            ensureVideoLoaded(video);
        });
        refreshManagedVideoPlayback();
        return;
    }

    videoLoadObserver = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                ensureVideoLoaded(entry.target);
            }
        });
    }, {
        root: null,
        rootMargin: '360px 0px',
        threshold: 0.01
    });

    videoVisibilityObserver = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            var video = entry.target;
            video.dataset.viewportVisible = entry.intersectionRatio >= 0.45 ? '1' : '0';
            updateVideoPlayback(video);
        });
    }, {
        root: null,
        threshold: [0, 0.2, 0.45, 0.8]
    });

    document.addEventListener('visibilitychange', refreshManagedVideoPlayback);
    window.addEventListener('focus', refreshManagedVideoPlayback);
    window.addEventListener('pageshow', refreshManagedVideoPlayback);

    var videos = document.querySelectorAll('video');
    videos.forEach(function(video) {
        registerManagedVideo(video);
    });
    bootstrapVideoVisibilityState();
    preloadActiveGalleryVideos(document.querySelector('.gallery-content.active'));
    refreshManagedVideoPlayback();
}

function registerManagedVideo(video) {
    if (!video || video.dataset.managedVideo === 'true') {
        return;
    }

    video.dataset.managedVideo = 'true';
    prepareMedia(video);
    if (!video.dataset.src && video.getAttribute('src')) {
        video.dataset.src = video.getAttribute('src');
    }

    var wrapper = video.closest('.video-comparison-wrap');
    if (wrapper) {
        ensureLoadingOverlay(wrapper);
    }

    video.addEventListener('canplay', function() {
        updateLoadingState(wrapper);
        updateVideoPlayback(video);
    });
    video.addEventListener('canplaythrough', function() {
        updateLoadingState(wrapper);
        updateVideoPlayback(video);
    });
    video.addEventListener('loadeddata', function() {
        updateLoadingState(wrapper);
        updateVideoPlayback(video);
    });
    video.addEventListener('waiting', function() {
        updateLoadingState(wrapper);
    });
    video.addEventListener('playing', function() {
        updateLoadingState(wrapper);
    });
    video.addEventListener('pause', function() {
        updateLoadingState(wrapper);
    });
    video.addEventListener('error', function() {
        updateLoadingState(wrapper);
    });

    if (videoLoadObserver) {
        videoLoadObserver.observe(video);
    }
    if (videoVisibilityObserver) {
        videoVisibilityObserver.observe(video);
    }
}

function ensureVideoLoaded(video) {
    if (!video || video.dataset.loaded === '1') {
        return;
    }

    var src = video.dataset.src || video.getAttribute('src');
    if (!src) {
        return;
    }

    if (video.getAttribute('src') !== src) {
        video.setAttribute('src', src);
    }
    video.preload = 'auto';
    video.load();
    video.dataset.loaded = '1';
}

function refreshManagedVideoPlayback() {
    var videos = document.querySelectorAll('video');
    videos.forEach(function(video) {
        registerManagedVideo(video);
        updateVideoPlayback(video);
    });
}

function canVideoAutoPlay(video) {
    if (!video || !video.isConnected || document.hidden) {
        return false;
    }
    var panel = video.closest('.gallery-content');
    if (panel && !panel.classList.contains('active')) {
        return false;
    }
    if (videoVisibilityObserver) {
        if (video.dataset.viewportVisible !== '1' && video.dataset.viewportVisible !== '0') {
            video.dataset.viewportVisible = isElementInViewport(video, 0.15) ? '1' : '0';
        }
    }
    if (videoVisibilityObserver && video.dataset.viewportVisible === '0' && !isElementInViewport(video, 0.08)) {
        return false;
    }
    if (!isElementVisible(video)) {
        return false;
    }
    return true;
}

function preloadActiveGalleryVideos(container) {
    if (!container) {
        return;
    }
    var videos = container.querySelectorAll('video');
    videos.forEach(function(video) {
        registerManagedVideo(video);
        ensureVideoLoaded(video);
    });
}

function bootstrapVideoVisibilityState() {
    var videos = document.querySelectorAll('video');
    videos.forEach(function(video) {
        if (video.dataset.viewportVisible === '1' || video.dataset.viewportVisible === '0') {
            return;
        }
        video.dataset.viewportVisible = isElementInViewport(video, 0.15) ? '1' : '0';
    });
}

function isElementInViewport(element, minRatio) {
    if (!element || element.getClientRects().length === 0) {
        return false;
    }
    var rect = element.getBoundingClientRect();
    var viewportW = window.innerWidth || document.documentElement.clientWidth;
    var viewportH = window.innerHeight || document.documentElement.clientHeight;
    var x = Math.max(0, Math.min(rect.right, viewportW) - Math.max(rect.left, 0));
    var y = Math.max(0, Math.min(rect.bottom, viewportH) - Math.max(rect.top, 0));
    var visibleArea = x * y;
    var totalArea = Math.max(1, rect.width * rect.height);
    return visibleArea / totalArea >= minRatio;
}

function isElementVisible(element) {
    if (!element || element.getClientRects().length === 0) {
        return false;
    }
    var style = window.getComputedStyle(element);
    return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
}

function updateVideoPlayback(video) {
    if (!video || !video.isConnected) {
        return;
    }

    if (!canVideoAutoPlay(video)) {
        if (!video.paused) {
            video.pause();
        }
        return;
    }

    ensureVideoLoaded(video);

    if (video.dataset.syncRole === 'follower') {
        return;
    }

    if (video.dataset.syncRole === 'leader' && typeof video._syncAlign === 'function') {
        video._syncAlign(false);
        return;
    }

    if (video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
        return;
    }

    if (video.paused) {
        var p = video.play();
        if (p && typeof p.catch === 'function') {
            p.catch(function() {});
        }
    }
}

function ensureLoadingOverlay(wrapper) {
    if (!wrapper || wrapper.querySelector('.video-loading-overlay')) {
        return;
    }

    var overlay = document.createElement('div');
    overlay.className = 'video-loading-overlay';
    overlay.innerHTML = '<span class="video-loading-spinner"></span><span class="video-loading-text">Loading...</span>';
    wrapper.appendChild(overlay);
}

function updateLoadingState(wrapper) {
    if (!wrapper) {
        return;
    }
    var videos = wrapper.querySelectorAll('video');
    if (!videos.length) {
        wrapper.classList.remove('is-loading');
        return;
    }

    var hasBlocking = false;
    videos.forEach(function(video) {
        if (video.error) {
            return;
        }
        if (!video.getAttribute('src') || video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
            hasBlocking = true;
        }
    });

    if (hasBlocking) {
        wrapper.classList.add('is-loading');
    } else {
        wrapper.classList.remove('is-loading');
    }
}
